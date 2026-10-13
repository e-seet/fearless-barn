#!/usr/bin/env python3
"""Experiment: does LightGBM's built-in `poisson` objective (log-link mean,
uniform gradient weighting -- same row-weighting as L2, unlike Huber/Tweedie/
Gamma) beat plain L2 on the heavy-tailed taxi-out target?

Why this variant and not log1p/Tweedie/Gamma (see tests_log_target/RESULT.md
and the Huber closure in TAXI_OUT_DRIVERS.md): those all mechanically
down-weight large-residual rows during training (gradient weight ~ mu^(1-p),
p>1), which is the same failure mode that made log1p score worse on pooled/
test RMSE despite looking better on val in-band. Poisson (p=1 on that same
scale) has *uniform* row weighting -- identical to L2 -- so it isolates a
different question: does the multiplicative/log-link mean function help,
independent of tail de-weighting? It also fits E[y|x] directly (no transform
to invert), so it does not have log1p's expm1/Jensen downward-bias problem.

Design mirrors tests_log_target/run_log_target.py exactly, minus the
transform machinery (poisson needs none):
* One script trains BOTH the L2 control and the poisson variant, identical
  data loading, identical base params, identical early-stopping cap/patience,
  identical output clipping.
* Seeds 42/43/44 (RETROSPECTIVE 4c: single-seed deltas are inside seed noise).
* Scored through common.evaluate.evaluate, same metric block every arm reports.
* Writes real run dirs under artifacts/runs/<arm>__<proto>__seed<S>__tag=<t>/
  (tags: l2ctl, poisson) so harvest_results.py can fold them into results.csv.

Protocol: forward / with_aobt / keep_all (the canonical one, matches
tests_log_target so the two experiments are directly comparable).

Pre-registered bar (decided before running, per project testing discipline):
poisson must beat the L2 control on BOTH test RMSE and test in-band by more
than the arm's seed-sd, AND the pooled test gain must clear ~5s to be worth
anything beyond a closed record on the objective-function axis.
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import evaluate as E
from common.bootstrap import run_dir, seed_everything
from pipeline_per_airport.model import PerAirportLGBM, AIRPORT_COL
from common import features as F

SPLIT, FEATURESET, POLICY = "forward", "with_aobt", "keep_all"
SEEDS = [42, 43, 44]
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC

BASE_PARAMS = {  # identical base hyperparams to pipeline_lightgbm/train.py PARAMS
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}
NUM_BOOST_ROUND, PATIENCE = 5000, 200

VARIANTS = {
    # matched objective/metric pair per variant, exactly like log1p used
    # log-space RMSE as its early-stopping metric.
    "l2ctl": {"objective": "l2", "metric": "rmse"},
    "poisson": {"objective": "poisson", "metric": "poisson"},
}


def _params(seed, variant):
    p = dict(BASE_PARAMS)
    p.update(VARIANTS[variant])
    p.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "device_type": "cpu"})
    return p


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def train_pooled(lgb, Xtr, ytr, Xva, yva, cats, seed, variant):
    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, reference=dtr,
                      free_raw_data=False)
    hist = {}
    booster = lgb.train(_params(seed, variant), dtr, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                                   lgb.record_evaluation(hist)])
    best = booster.best_iteration or NUM_BOOST_ROUND
    hit_cap = bool(best >= NUM_BOOST_ROUND)
    conv = {"best_iteration": int(best), "max_rounds": NUM_BOOST_ROUND,
            "patience": PATIENCE, "early_stop_fired": not hit_cap,
            "hit_max_rounds": hit_cap}
    return booster, best, conv


def predict_pooled(booster, X, best):
    return clip(booster.predict(X, num_iteration=best))


def write_run(arm, seed, tag, conv, res_val, res_test):
    proto = config.protocol_key(SPLIT, FEATURESET, POLICY)
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed,
                extra={"tag": tag})
    (d / "model.json").write_text(json.dumps({
        "arm": arm, "split": SPLIT, "featureset": FEATURESET, "seed": seed,
        "tag": tag, "objective": VARIANTS[tag]["objective"],
        "params": _params(seed, tag), "convergence": conv,
        "selection_surface": "val",
        "note": "tests_poisson_objective/run_poisson.py -- poisson-objective "
                "experiment; L2 control and poisson variant trained by the same "
                "script",
    }, indent=2))
    (d / "eval_val.json").write_text(json.dumps(res_val, indent=2))
    (d / "eval_test.json").write_text(json.dumps(res_test, indent=2))
    return d


def main() -> int:
    import lightgbm as lgb
    t0 = time.time()
    rows = []

    # ---- load once per split part (seed-independent) ----
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
    assert (ytr >= 0).all(), "poisson needs non-negative targets; found <0 in train"
    print(f"loaded: train={len(Xtr_raw)} val={len(Xva)} test={len(Xte)} "
          f"cats={len(cats)}  ({time.time()-t0:.0f}s)")

    def score(arm, seed, tag, p_val, p_test, conv):
        rv = E.evaluate(yva, p_val, meta_va, ytr, part="val", arm=arm, split=SPLIT,
                        featureset=FEATURESET, seed=seed)
        rt = E.evaluate(yte, p_test, meta_te, ytr, part="test", arm=arm, split=SPLIT,
                        featureset=FEATURESET, seed=seed)
        write_run(arm, seed, tag, conv, rv, rt)
        row = {"arm": arm, "seed": seed, "variant": tag,
               "val_rmse": round(rv["rmse"], 3), "val_inband": round(rv["rmse_in_band"], 3),
               "test_rmse": round(rt["rmse"], 3), "test_inband": round(rt["rmse_in_band"], 3),
               "test_lb_weighted": round(rt.get("leaderboard_weighted_rmse", float("nan")), 3),
               "best_iter": conv.get("best_iteration") if isinstance(conv, dict) else conv,
               "hit_cap": conv.get("hit_max_rounds") if isinstance(conv, dict) else None}
        rows.append(row)
        print(f"  {arm:12s} seed{seed} {tag:8s}  "
              f"val {row['val_rmse']:8.2f} / ib {row['val_inband']:7.2f}   "
              f"test {row['test_rmse']:8.2f} / ib {row['test_inband']:7.2f} / lbw {row['test_lb_weighted']:7.2f}"
              f"  (best_iter {row['best_iter']})")

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n===== seed {seed} =====")

        # ---- pooled lightgbm ----
        for tag in ["l2ctl", "poisson"]:
            b, best, conv = train_pooled(lgb, Xtr_raw, ytr, Xva, yva, cats, seed, tag)
            score("lightgbm", seed, tag,
                  predict_pooled(b, Xva, best),
                  predict_pooled(b, Xte, best), conv)

        # ---- per_airport ----
        for tag in ["l2ctl", "poisson"]:
            m = PerAirportLGBM(_params(seed, tag), NUM_BOOST_ROUND, PATIENCE, cats)
            m.fit(Xtr_raw, ytr, pool_meta, Xva, yva, meta_va, lgb=lgb, verbose=False)
            pv = clip(m.predict(Xva, meta_va))
            pt = clip(m.predict(Xte, meta_te))
            conv = {"best_iteration": {k: int(v) for k, v in m.best_iter_.items()},
                    "any_hit_cap": any(c["hit_max_rounds"] for c in m.convergence_.values())}
            score("per_airport", seed, tag, pv, pt, conv)

    # ---- summary csv + prose ----
    with (HERE / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def agg(arm, variant, key):
        xs = [r[key] for r in rows if r["arm"] == arm and r["variant"] == variant]
        return (float(np.mean(xs)), float(np.std(xs)), xs)

    lines = ["poisson-objective experiment  (forward / with_aobt / keep_all, seeds 42/43/44)",
             "=" * 78,
             f"{'arm / metric':26s} {'L2 mean':>10s} {'poisson mean':>13s} {'delta':>9s}  "
             f"{'L2 seed-sd':>11s}  verdict"]
    for arm in ["lightgbm", "per_airport"]:
        for key, label in [("val_rmse", "val RMSE"), ("val_inband", "val in-band"),
                           ("test_rmse", "test RMSE"), ("test_inband", "test in-band"),
                           ("test_lb_weighted", "test LB-weighted")]:
            l2m, l2sd, _ = agg(arm, "l2ctl", key)
            pom, posd, _ = agg(arm, "poisson", key)
            delta = pom - l2m
            better = delta < 0
            decisive = abs(delta) > max(l2sd, posd)  # bigger than the seed spread?
            verdict = ("poisson better" if better else "poisson worse")
            verdict += " (decisive)" if decisive else " (within seed noise)"
            lines.append(f"{arm+' '+label:26s} {l2m:10.2f} {pom:13.2f} {delta:+9.2f}  "
                         f"{l2sd:11.2f}  {verdict}")
        lines.append("")
    txt = "\n".join(lines)
    (HERE / "summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n[{time.time()-t0:.0f}s]  wrote {HERE/'summary.csv'} and summary.txt; "
          f"run dirs under artifacts/runs/*__tag=l2ctl|poisson")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
