#!/usr/bin/env python3
"""Experiment: a matched/unmatched ROUTER -- two separate LightGBM specialists
(one trained only on rows with an NM flight record, one trained only on rows
without) -- versus one pooled LightGBM trained on everything with the flag as
a feature.

Why this and not a rerun of something closed. This project has closed three
prior attempts on the AOBT-unmatched population (TAXI_OUT_DRIVERS.md sec.2q):
the blend-weight solver's own [0,0,1] result, an OpenAir-style proxy
override, and a single global-weight convex blend of the MLP prediction and
`delta_sched_sec`. All three were LIRF-only (283 rows, forward val) and all
three combined two existing POINT PREDICTIONS after the fact -- none trained
a model with the full feature set on the unmatched population itself, and
none used the pooled cross-airport population (the full unmatched pool is
~1.08% of all training rows, an order of magnitude more rows than LIRF's
283-row slice alone). That is what this script tests: a genuinely different
architecture (two specialist learners, not one point-estimate blend), on a
genuinely different population (all 10 airports' unmatched rows pooled, not
LIRF alone).

Prior art this DOES already anticipate failing the same way: sec.2y found
per-AIRPORT specialization makes 9/10 airports worse, because pooling shares
statistical strength (schedule/congestion patterns) that a narrow specialist
loses. A matched/unmatched split faces the identical risk on the smaller
side: ~1% of rows is a much smaller training pool for the unmatched
specialist to learn from, with less within-population structure to fall back
on. That's the actual empirical question here, not assumed either way.

Design mirrors tests_poisson_objective/run_poisson.py: one script, both the
pooled L2 control and the router variant, identical data/params/early
stopping, seeds 42/43/44, forward/with_aobt/keep_all, scored through
common.evaluate.evaluate.

Pre-registered bar (decided before running):
1. Router must beat the pooled control on POOLED test RMSE by more than the
   control's own seed-sd, to count as a real gain worth building.
2. Separately (regardless of #1): does the router at least help the
   UNMATCHED stratum specifically, even if it's a wash or slightly negative
   pooled (since the unmatched stratum is ~1% of rows, a real per-stratum WIN
   there could be swamped in the pooled number but still worth combining with
   the pooled model via a flag-gated ensemble)?
3. The decorrelation gate (only meaningful if #1 passes): corr(residual of a
   router-blended prediction, residual of the shipped airport_operator_multimodel
   blend) on `forward` val+test. If a would-be improvement's residual still
   correlates ~0.87 like every other closed "new arm" attempt (sec.2t/2u),
   a per-airport simplex re-solve will hand it ~0 weight and it is not worth
   building further regardless of #1.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import evaluate as E
from common import features as F
from common.bootstrap import run_dir, seed_everything
from common.predict import predict_arm

SPLIT, FEATURESET, POLICY = "forward", "with_aobt", "keep_all"
SEEDS = [42, 43, 44]
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC

BASE_PARAMS = {  # identical to pipeline_lightgbm/train.py PARAMS
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}
NUM_BOOST_ROUND, PATIENCE = 5000, 200
# the unmatched stratum is tiny (~1% of rows); a smaller leaf floor lets it
# actually build splits instead of degenerating to a near-constant predictor
UNMATCHED_MIN_DATA_IN_LEAF = 20


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype="float64") - p) ** 2)))


def _params(seed, min_data_in_leaf=None):
    p = dict(BASE_PARAMS)
    p.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "device_type": "cpu"})
    if min_data_in_leaf is not None:
        p["min_data_in_leaf"] = min_data_in_leaf
    return p


def train_one(lgb, Xtr, ytr, Xva, yva, cats, seed, min_data_in_leaf=None):
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr,
                      free_raw_data=False)
    hist = {}
    booster = lgb.train(_params(seed, min_data_in_leaf), dtr,
                        num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                                   lgb.record_evaluation(hist)])
    best = booster.best_iteration or NUM_BOOST_ROUND
    hit_cap = bool(best >= NUM_BOOST_ROUND)
    conv = {"best_iteration": int(best), "max_rounds": NUM_BOOST_ROUND,
            "patience": PATIENCE, "early_stop_fired": not hit_cap,
            "hit_max_rounds": hit_cap, "n_train": int(len(Xtr))}
    return booster, best, conv


def write_run(arm, seed, tag, conv, res_val, res_test):
    proto = config.protocol_key(SPLIT, FEATURESET, POLICY)
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed,
                extra={"tag": tag})
    (d / "model.json").write_text(json.dumps({
        "arm": arm, "split": SPLIT, "featureset": FEATURESET, "seed": seed,
        "tag": tag, "params": {"objective": "l2"}, "convergence": conv,
        "selection_surface": "val",
        "note": "tests_matched_unmatched_router/run_router.py -- pooled-vs-router "
                "experiment, diagnostic only, not a registered arm",
    }, indent=2))
    (d / "eval_val.json").write_text(json.dumps(res_val, indent=2))
    (d / "eval_test.json").write_text(json.dumps(res_test, indent=2))
    return d


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()
    rows = []

    Xtr_raw, ytr, pool_meta = D.load_part(SPLIT, "train", FEATURESET,
                                          filter_target=True, outlier_policy=POLICY)
    Xva_raw, yva, meta_va = D.load_part(SPLIT, "val", FEATURESET, filter_target=False)
    Xte_raw, yte, meta_te = D.load_part(SPLIT, "test", FEATURESET, filter_target=False)
    Xva = D.align_columns(Xtr_raw, Xva_raw)
    Xte = D.align_columns(Xtr_raw, Xte_raw)
    cats = [c for c in F.CATEGORICAL if c in Xtr_raw.columns]
    ytr = np.asarray(ytr, dtype="float64")
    yva = np.asarray(yva, dtype="float64")
    yte = np.asarray(yte, dtype="float64")

    m_tr = Xtr_raw["has_aobt"].to_numpy().astype(bool)
    m_va = Xva["has_aobt"].to_numpy().astype(bool)
    m_te = Xte["has_aobt"].to_numpy().astype(bool)
    print(f"loaded: train={len(Xtr_raw)} (unmatched={int((~m_tr).sum())}, "
          f"{(~m_tr).mean():.3%})  val={len(Xva)} (unmatched {int((~m_va).sum())}, "
          f"{(~m_va).mean():.3%})  test={len(Xte)} (unmatched {int((~m_te).sum())}, "
          f"{(~m_te).mean():.3%})  ({time.time()-t0:.0f}s)")
    if (~m_tr).sum() < 500:
        print("FEASIBILITY FAIL: unmatched training pool under 500 rows -- "
              "aborting before training anything.")
        return 1

    def score(arm, seed, tag, p_val, p_test, conv):
        rv = E.evaluate(yva, p_val, meta_va, ytr, part="val", arm=arm, split=SPLIT,
                        featureset=FEATURESET, seed=seed)
        rt = E.evaluate(yte, p_test, meta_te, ytr, part="test", arm=arm, split=SPLIT,
                        featureset=FEATURESET, seed=seed)
        write_run(arm, seed, tag, conv, rv, rt)
        row = {"arm": arm, "seed": seed, "variant": tag,
               "val_rmse": round(rv["rmse"], 3), "val_inband": round(rv["rmse_in_band"], 3),
               "test_rmse": round(rt["rmse"], 3), "test_inband": round(rt["rmse_in_band"], 3),
               "val_rmse_matched": round(rmse(yva[m_va], p_val[m_va]), 3),
               "val_rmse_unmatched": round(rmse(yva[~m_va], p_val[~m_va]), 3),
               "test_rmse_matched": round(rmse(yte[m_te], p_test[m_te]), 3),
               "test_rmse_unmatched": round(rmse(yte[~m_te], p_test[~m_te]), 3)}
        rows.append(row)
        print(f"  {arm:10s} seed{seed} {tag:10s}  val {row['val_rmse']:8.2f} "
              f"(m {row['val_rmse_matched']:7.2f} / u {row['val_rmse_unmatched']:9.2f})   "
              f"test {row['test_rmse']:8.2f} (m {row['test_rmse_matched']:7.2f} / "
              f"u {row['test_rmse_unmatched']:9.2f})")
        return p_val, p_test

    resid_val = {}
    resid_test = {}
    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n===== seed {seed} =====")

        # ---- pooled control (identical recipe to the registered l2ctl arm) ----
        b, best, conv = train_one(lgb, Xtr_raw, ytr, Xva, yva, cats, seed)
        p_val_pool = clip(b.predict(Xva, num_iteration=best))
        p_test_pool = clip(b.predict(Xte, num_iteration=best))
        pv, pt = score("lightgbm", seed, "pool_ctl", p_val_pool, p_test_pool, conv)
        resid_val["pool"] = yva - pv
        resid_test["pool"] = yte - pt

        # ---- router: two independently-trained specialists ----
        bm, bestm, convm = train_one(lgb, Xtr_raw[m_tr], ytr[m_tr],
                                     Xva[m_va], yva[m_va], cats, seed)
        bu, bestu, convu = train_one(lgb, Xtr_raw[~m_tr], ytr[~m_tr],
                                     Xva[~m_va], yva[~m_va], cats, seed,
                                     min_data_in_leaf=UNMATCHED_MIN_DATA_IN_LEAF)
        p_val_router = np.empty(len(yva), dtype="float64")
        p_val_router[m_va] = clip(bm.predict(Xva[m_va], num_iteration=bestm))
        p_val_router[~m_va] = clip(bu.predict(Xva[~m_va], num_iteration=bestu))
        p_test_router = np.empty(len(yte), dtype="float64")
        p_test_router[m_te] = clip(bm.predict(Xte[m_te], num_iteration=bestm))
        p_test_router[~m_te] = clip(bu.predict(Xte[~m_te], num_iteration=bestu))
        conv_router = {"matched": convm, "unmatched": convu}
        pv, pt = score("router", seed, "router", p_val_router, p_test_router, conv_router)
        resid_val["router"] = yva - pv
        resid_test["router"] = yte - pt

    # ---- decorrelation gate: only meaningful for the LAST seed's residuals,
    # against the shipped blend on the SAME forward val+test rows ----
    print("\n===== decorrelation gate (seed {}, forward val+test) =====".format(SEEDS[-1]))
    p_shipped_val = predict_arm("airport_operator_multimodel", SPLIT, "val", FEATURESET,
                                SEEDS[-1], POLICY, X=Xva, meta=meta_va, tag="msmlp")
    p_shipped_test = predict_arm("airport_operator_multimodel", SPLIT, "test", FEATURESET,
                                 SEEDS[-1], POLICY, X=Xte, meta=meta_te, tag="msmlp")
    resid_shipped_val = yva - p_shipped_val
    resid_shipped_test = yte - p_shipped_test
    corr_val = float(np.corrcoef(resid_val["router"], resid_shipped_val)[0, 1])
    corr_test = float(np.corrcoef(resid_test["router"], resid_shipped_test)[0, 1])
    print(f"  corr(router resid, shipped blend resid): val={corr_val:.4f}  test={corr_test:.4f}")
    print(f"  (bar: <0.70 to be worth a simplex re-solve; ~0.87 means dead on arrival)")

    with (HERE / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def agg(arm, variant, key):
        xs = [r[key] for r in rows if r["arm"] == arm and r["variant"] == variant]
        return (float(np.mean(xs)), float(np.std(xs)), xs)

    lines = ["matched/unmatched router experiment  "
             "(forward / with_aobt / keep_all, seeds 42/43/44)",
             "=" * 78]
    for key, label in [("val_rmse", "val RMSE pooled"), ("val_rmse_matched", "val RMSE matched"),
                       ("val_rmse_unmatched", "val RMSE unmatched"),
                       ("test_rmse", "test RMSE pooled"), ("test_rmse_matched", "test RMSE matched"),
                       ("test_rmse_unmatched", "test RMSE unmatched")]:
        cm, csd, _ = agg("lightgbm", "pool_ctl", key)
        rm_, rsd, _ = agg("router", "router", key)
        delta = rm_ - cm
        decisive = abs(delta) > max(csd, rsd)
        verdict = ("router better" if delta < 0 else "router worse")
        verdict += " (decisive)" if decisive else " (within seed noise)"
        lines.append(f"{label:20s} pool {cm:10.2f} (sd {csd:5.2f})  router {rm_:10.2f} "
                     f"(sd {rsd:5.2f})  delta {delta:+8.2f}  {verdict}")
    lines.append("")
    lines.append(f"decorrelation gate (seed {SEEDS[-1]}): corr val={corr_val:.4f} test={corr_test:.4f}")
    txt = "\n".join(lines)
    (HERE / "summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n[{time.time()-t0:.0f}s]  wrote {HERE/'summary.csv'} and summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
