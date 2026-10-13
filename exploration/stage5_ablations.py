#!/usr/bin/env python3
"""Stage 5 -- confound checks on the arm comparison itself.

Design-gate finding S10. Under the `forward` split the model trains on Jan-Aug
and is tested on Nov-Dec, so `month_num` and `doy` are OUT OF RANGE on every test
row. A GBM drops them into August's rightmost bin; the ridge extrapolates
smoothly through `doy_sin`/`doy_cos`. That is a property of the split, not of the
method -- and it sits directly on the axis the arms are ranked on. It also
vanishes in the real task, where the 12-month refit has seen every month.

This stage refits each arm with `ABSOLUTE_TIME_COLS` dropped and checks the arm
RANKING is unchanged. A changed ranking means the comparison was measuring
extrapolation behaviour, not method quality.

Output: artifacts/ablations/<protocol>/ablation_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common import features as F
from common.metrics import rmse
from common.bootstrap import run_dir, seed_everything, write_json_guarded

FAST = {"objective": "l2", "metric": "rmse", "learning_rate": 0.1,
        "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb
    from sklearn.linear_model import Ridge  # noqa: F401  (import cost check)
    from pipeline_ridge.train import RidgePipeline

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "ablations", "ablation", protocol=proto, seed=args.seed)

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
    Xva = D.align_columns(Xtr, Xva)
    yva_np = np.asarray(yva, dtype="float64")

    results = {}
    for variant, drop in (("with_absolute_time", []),
                          ("without_absolute_time", F.ABSOLUTE_TIME_COLS)):
        Xt = Xtr.drop(columns=[c for c in drop if c in Xtr.columns])
        Xv = Xva.drop(columns=[c for c in drop if c in Xva.columns])
        cats = [c for c in F.CATEGORICAL if c in Xt.columns]

        d = lgb.Dataset(Xt, label=np.asarray(ytr, dtype="float64"),
                        categorical_feature=cats, free_raw_data=False)
        b = lgb.train(dict(FAST, seed=args.seed), d, num_boost_round=args.rounds)
        gbm = rmse(yva_np, np.clip(b.predict(Xv), config.PREDICTION_CLIP_MIN_SEC,
                                   config.PREDICTION_CLIP_MAX_SEC))
        rp = RidgePipeline(args.seed, 10.0).fit(Xt, ytr)
        rdg = rmse(yva_np, rp.predict(Xv))

        results[variant] = {"lightgbm_val_rmse": gbm, "ridge_val_rmse": rdg,
                            "gap_ridge_minus_gbm": rdg - gbm,
                            "winner": "lightgbm" if gbm < rdg else "ridge",
                            "n_features": int(Xt.shape[1])}
        print("  {:24s} lgbm={:8.3f} ridge={:8.3f} gap={:+8.3f} winner={}".format(
            variant, gbm, rdg, rdg - gbm, results[variant]["winner"]))

    same = results["with_absolute_time"]["winner"] == results["without_absolute_time"]["winner"]
    print("\n  arm ranking unchanged by dropping absolute-time features: {}".format(same))
    if not same:
        print("  WARNING: the arm ranking DEPENDS on the absolute-time features, "
              "which are out of range on this split's test months by construction. "
              "The comparison is measuring extrapolation behaviour, not method "
              "quality -- do not register a winner from it.")

    rep = {"split": args.split, "featureset": args.featureset, "seed": args.seed,
           "outlier_policy": args.outlier_policy,
           "absolute_time_cols": list(F.ABSOLUTE_TIME_COLS),
           "results": results, "ranking_unchanged": bool(same), "passed": bool(same)}
    write_json_guarded(out / "ablation_report.json", rep, force=args.force)
    print("  report: {}".format(out / "ablation_report.json"))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
