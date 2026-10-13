#!/usr/bin/env python3
"""Experiment: does a log1p(target) transform help on the heavy-tailed taxi-out
target?  Tested on the two cleanest arms -- pooled `lightgbm` and `per_airport`.

Design
------
* One script trains BOTH the L2 control and the log1p variant, with identical
  data loading, identical params, identical early-stopping config and identical
  output clipping -- so the comparison is not confounded by code drift
  (RETROSPECTIVE MISTAKES #8: a mixed-feature-code comparison once crowned a
  wrong winner here).
* log1p variant: label := log1p(y); early stopping is on val RMSE *in log space*
  (matched objective); predictions are expm1'd then clipped to [60, 86400] s
  exactly like every other arm.
* Seeds 42/43/44, because a single-seed delta on this metric is inside the seed
  spread (RETROSPECTIVE 4c).
* Scored through `common.evaluate.evaluate` -- the same metric block every arm
  reports, so `rmse` / `rmse_in_band` are comparable at face value.
* Writes real run dirs under artifacts/runs/<arm>__<proto>__seed<S>__tag=<t>/
  (tags: `l2ctl`, `log1p`) so `harvest_results.py` folds them into results.csv.

Protocol: forward / with_aobt / keep_all  (the canonical one).
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
from common.bootstrap import run_dir, seed_everything
from pipeline_per_airport.model import PerAirportLGBM, AIRPORT_COL
from common import features as F

SPLIT, FEATURESET, POLICY = "forward", "with_aobt", "keep_all"
SEEDS = [42, 43, 44]
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC

PARAMS = {  # identical to pipeline_lightgbm/train.py PARAMS
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}
NUM_BOOST_ROUND, PATIENCE = 5000, 200


def _params(seed):
    p = dict(PARAMS)
    p.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed,
              "device_type": "cpu"})
    return p


def fwd(y, use_log):
    """Forward target transform. For log1p, floor at 1 s first: ~0.01% of val/test
    rows carry a negative sentinel target (min -12 s) that the training filter
    (y>=1) already excludes, and log1p of those is -inf/NaN. Flooring only affects
    the log-space early-stopping label; scoring always uses the raw y, and
    predictions are expm1'd then clipped to [60, 86400] like every arm."""
    y = np.asarray(y, dtype="float64")
    if not use_log:
        return y
    return np.log1p(np.maximum(y, 1.0))


def inv(p, use_log):
    p = np.asarray(p, dtype="float64")
    return np.expm1(p) if use_log else p


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def train_pooled(lgb, Xtr, ytr, Xva, yva, cats, seed, use_log):
    dtr = lgb.Dataset(Xtr, label=fwd(ytr, use_log), categorical_feature=cats,
                      free_raw_data=False)
    dva = lgb.Dataset(Xva, label=fwd(yva, use_log), categorical_feature=cats,
                      reference=dtr, free_raw_data=False)
    hist = {}
    booster = lgb.train(_params(seed), dtr, num_boost_round=NUM_BOOST_ROUND,
                        valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(PATIENCE, verbose=False),
                                   lgb.record_evaluation(hist)])
    best = booster.best_iteration or NUM_BOOST_ROUND
    hit_cap = bool(best >= NUM_BOOST_ROUND)
    conv = {"best_iteration": int(best), "max_rounds": NUM_BOOST_ROUND,
            "patience": PATIENCE, "early_stop_fired": not hit_cap,
            "hit_max_rounds": hit_cap}
    return booster, best, conv


def predict_pooled(booster, X, best, use_log):
    return clip(inv(booster.predict(X, num_iteration=best), use_log))


def write_run(arm, seed, tag, conv, res_val, res_test):
    proto = config.protocol_key(SPLIT, FEATURESET, POLICY)
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed,
                extra={"tag": tag})
    (d / "model.json").write_text(json.dumps({
        "arm": arm, "split": SPLIT, "featureset": FEATURESET, "seed": seed,
        "tag": tag, "target_transform": ("log1p" if tag == "log1p" else "identity"),
        "params": _params(seed), "convergence": conv,
        "selection_surface": "val",
        "note": "tests_log_target/run_log_target.py -- log-target experiment; "
                "L2 control and log1p variant trained by the same script",
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
    assert (ytr > 0).all(), "log1p needs positive targets; found <=0 in train"
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
        print(f"  {arm:12s} seed{seed} {tag:6s}  "
              f"val {row['val_rmse']:8.2f} / ib {row['val_inband']:7.2f}   "
              f"test {row['test_rmse']:8.2f} / ib {row['test_inband']:7.2f} / lbw {row['test_lb_weighted']:7.2f}"
              f"  (best_iter {row['best_iter']})")

    for seed in SEEDS:
        seed_everything(seed)
        print(f"\n===== seed {seed} =====")

        # ---- pooled lightgbm ----
        for use_log, tag in [(False, "l2ctl"), (True, "log1p")]:
            b, best, conv = train_pooled(lgb, Xtr_raw, ytr, Xva, yva, cats, seed, use_log)
            score("lightgbm", seed, tag,
                  predict_pooled(b, Xva, best, use_log),
                  predict_pooled(b, Xte, best, use_log), conv)

        # ---- per_airport ----
        for use_log, tag in [(False, "l2ctl"), (True, "log1p")]:
            m = PerAirportLGBM(_params(seed), NUM_BOOST_ROUND, PATIENCE, cats)
            m.fit(Xtr_raw, fwd(ytr, use_log), pool_meta, Xva, fwd(yva, use_log),
                  meta_va, lgb=lgb, verbose=False)
            pv = clip(inv(m.predict(Xva, meta_va), use_log))
            pt = clip(inv(m.predict(Xte, meta_te), use_log))
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

    lines = ["log-target experiment  (forward / with_aobt / keep_all, seeds 42/43/44)",
             "=" * 78,
             f"{'arm / metric':26s} {'L2 mean':>10s} {'log1p mean':>12s} {'delta':>9s}  "
             f"{'L2 seed-sd':>11s}  verdict"]
    for arm in ["lightgbm", "per_airport"]:
        for key, label in [("val_rmse", "val RMSE"), ("val_inband", "val in-band"),
                           ("test_rmse", "test RMSE"), ("test_inband", "test in-band"),
                           ("test_lb_weighted", "test LB-weighted")]:
            l2m, l2sd, _ = agg(arm, "l2ctl", key)
            lgm, lgsd, _ = agg(arm, "log1p", key)
            delta = lgm - l2m
            better = delta < 0
            decisive = abs(delta) > max(l2sd, lgsd)  # bigger than the seed spread?
            verdict = ("log1p better" if better else "log1p worse")
            verdict += " (decisive)" if decisive else " (within seed noise)"
            lines.append(f"{arm+' '+label:26s} {l2m:10.2f} {lgm:12.2f} {delta:+9.2f}  "
                         f"{l2sd:11.2f}  {verdict}")
        lines.append("")
    txt = "\n".join(lines)
    (HERE / "summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n[{time.time()-t0:.0f}s]  wrote {HERE/'summary.csv'} and summary.txt; "
          f"run dirs under artifacts/runs/*__tag=l2ctl|log1p")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
