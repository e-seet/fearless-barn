#!/usr/bin/env python3
"""Training and per-airport meta-weight solving for airport_catboost_multimodel.

Solves per-airport convex blend weights combining:
1. catboost_antigravity (global CatBoost, ordered-target-statistics categoricals;
   in-band val RMSE ~234.8, essentially tied with the MLP in-band, but a much
   worse tail -- see the arm's own docstring)
2. mlp_antigravity (the best single arm overall; val ~253.99)

Direct copy of pipeline_airport_operator_multimodel/train.py's structure and
discipline -- same reason: this is the cheap, direct test of "does swapping
in a differently-biased tree-side parent help," just with catboost_antigravity
in place of airport_operator_blend. Weights are solved on forward_inner (a
holdout carved from TRAIN, never val), same reasoning: MISTAKES.md's lever-3
lesson (in-sample weights measured 246.94, honest out-of-sample 254.24/252.68 --
WORSE than baseline) is a direct warning against reading a blend weight off
val itself. This arm additionally requires catboost_antigravity to itself be
trained under the forward_inner protocol first (same rule
pipeline_airport_operator_multimodel imposes on airport_operator_blend), so
predict_arm("catboost_antigravity", "forward_inner", "val", ...) is not
leaking predictions from a model that saw the held-out month.

A CatBoost-side seed loop is available (--catboost-seeds) for the same reason
mlp_prediction averages the MLP: if CatBoost's own seed variance turns out to
matter, averaging is zero-fitted-parameter variance reduction. Off (single
seed) by default since this is a first honest look, not yet a shipped config.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_airport_operator_multimodel.train import (parse_seeds, mlp_prediction,
                                                         solve_alpha_per_airport)

ARM = "airport_catboost_multimodel"


def catboost_prediction(split, part, featureset, seeds, outlier_policy, X, meta, Xtr,
                        tag=None):
    preds = [predict_arm("catboost_antigravity", split, part, featureset, s,
                         outlier_policy, X=X, meta=meta, Xtr=Xtr, tag=tag)
             for s in seeds]
    return np.mean(preds, axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--mlp-tag", default=None)
    ap.add_argument("--catboost-tag", default=None)
    ap.add_argument("--mlp-seeds", default="",
                    help="comma-separated seeds of mlp_antigravity to AVERAGE. "
                         "Default empty = use --seed alone.")
    ap.add_argument("--catboost-seeds", default="",
                    help="comma-separated seeds of catboost_antigravity to AVERAGE. "
                         "Default empty = use --seed alone.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--weight-split", default="forward_inner", choices=["forward_inner", "val"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    mlp_seeds = parse_seeds(args.mlp_seeds, args.seed)
    cb_seeds = parse_seeds(args.catboost_seeds, args.seed)
    print(f"[{ARM}] MLP seeds: {mlp_seeds}  CatBoost seeds: {cb_seeds}")

    Xtr, _, _ = D.load_part(args.split, "train", args.featureset, filter_target=True,
                            outlier_policy=args.outlier_policy)

    def solve_on(wsplit):
        Xw_tr, _, _ = D.load_part(wsplit, "train", args.featureset, filter_target=True,
                                  outlier_policy=args.outlier_policy)
        Xw, yw, meta_w = D.load_part(wsplit, "val", args.featureset, filter_target=False)
        yw = np.asarray(yw, dtype="float64")
        pa_w = catboost_prediction(wsplit, "val", args.featureset, cb_seeds,
                                   args.outlier_policy, Xw, meta_w, Xw_tr,
                                   tag=args.catboost_tag)
        pb_w = mlp_prediction(wsplit, "val", args.featureset, mlp_seeds,
                              args.outlier_policy, Xw, meta_w, Xw_tr,
                              tag=args.mlp_tag)
        ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()
        return solve_alpha_per_airport(yw, pa_w, pb_w, ap_w)

    weights_per_airport = solve_on("forward_inner" if args.weight_split == "forward_inner"
                                   else args.split)

    print(f"[{ARM}] Solved weights (weight on catboost_antigravity):")
    for k, v in weights_per_airport.items():
        print(f"  {k}: {v:.4f} (MLP weight: {1.0 - v:.4f})")

    has_val = bool(config.SPLITS[args.split]["val"])
    val_rmse = None
    val_rmse_cb_alone = None
    val_rmse_mlp_alone = None
    if has_val:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pa_val = catboost_prediction(args.split, "val", args.featureset, cb_seeds,
                                     args.outlier_policy, Xva, meta_va, Xtr,
                                     tag=args.catboost_tag)
        pb_val = mlp_prediction(args.split, "val", args.featureset, mlp_seeds,
                                args.outlier_policy, Xva, meta_va, Xtr,
                                tag=args.mlp_tag)
        ap_val = meta_va[AIRPORT_COL].astype("object").to_numpy()
        p_val_blend = np.zeros(len(yva), dtype="float64")
        for a in pd.unique(ap_val):
            m = (ap_val == a)
            w = weights_per_airport.get(str(a), 0.5)
            p_val_blend[m] = w * pa_val[m] + (1.0 - w) * pb_val[m]
        p_val_blend = np.clip(p_val_blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, p_val_blend)
        val_rmse_cb_alone = rmse(yva, np.clip(pa_val, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC))
        val_rmse_mlp_alone = rmse(yva, np.clip(pb_val, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC))
        print(f"[{ARM}] Forward Validation Blend RMSE: {val_rmse:.4f}  "
              f"[catboost alone {val_rmse_cb_alone:.4f} | mlp alone {val_rmse_mlp_alone:.4f}]")

    sel = declare_selection_set(f"{ARM} per-airport alpha",
                                selected_on=args.weight_split, value=weights_per_airport)

    meta = {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "outlier_policy": args.outlier_policy, "seed": args.seed,
        "weights_per_airport": weights_per_airport,
        "weight_split": args.weight_split,
        "mlp_seeds": mlp_seeds, "catboost_seeds": cb_seeds,
        "catboost_tag": args.catboost_tag, "mlp_tag": args.mlp_tag,
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        "val_rmse_catboost_alone": float(val_rmse_cb_alone) if val_rmse_cb_alone is not None else None,
        "val_rmse_mlp_alone": float(val_rmse_mlp_alone) if val_rmse_mlp_alone is not None else None,
        "selection": dict(sel),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] Meta-model metadata saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
