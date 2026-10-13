#!/usr/bin/env python3
"""Training and meta-weight solving for pipeline_multimodel_antigravity.

Solves per-airport convex blend weights combining:
1. per_airport (Specialist LightGBM per airport)
2. mlp_antigravity (Tabular neural net with entity embeddings)

Weights can be solved on forward_inner (honest holdout) or on validation.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
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

ARM = "multimodel_antigravity"


def parse_seeds(spec: str, default_seed: int) -> "list[int]":
    """'' -> [default_seed]; '42,43,44' -> [42, 43, 44]."""
    if not spec:
        return [int(default_seed)]
    return [int(x) for x in str(spec).split(",") if str(x).strip() != ""]


def mlp_prediction(split, part, featureset, seeds, outlier_policy, X, meta, Xtr,
                   tag=None):
    """mlp_antigravity's prediction, averaged over `seeds`. Single seed = mean of one.

    Same rationale as airport_operator_multimodel.mlp_prediction: the tree side
    (per_airport) is near-deterministic across seeds; essentially all of this
    blend's seed variance enters through the MLP, so averaging independent draws
    of the noisy parent is zero-parameter variance reduction.
    """
    preds = [predict_arm("mlp_antigravity", split, part, featureset, s, outlier_policy,
                         X=X, meta=meta, Xtr=Xtr, tag=tag)
             for s in seeds]
    return np.mean(preds, axis=0)


def solve_alpha_per_airport(y, pa, pb, ap) -> dict[str, float]:
    """Solve least squares weight alpha in [0, 1] per airport: y ~ a*pa + (1-a)*pb."""
    weights = {}
    for a in sorted(pd.unique(ap)):
        m = (ap == a)
        ya = y[m]
        pa_m = pa[m]
        pb_m = pb[m]
        d = pa_m - pb_m
        r = ya - pb_m
        denom = float(np.dot(d, d))
        if denom <= 1e-12:
            alpha = 0.5
        else:
            alpha = float(np.dot(d, r) / denom)
        weights[str(a)] = float(np.clip(alpha, 0.0, 1.0))
    return weights


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--mlp-seeds", default="",
                    help="Comma-separated seeds of mlp_antigravity to AVERAGE for "
                         "the neural side (e.g. 42,43,44). Default empty = use "
                         "--seed alone, i.e. the original single-draw behaviour.")
    ap.add_argument("--tag", default=None,
                    help="Run-dir tag, so an averaged variant never overwrites "
                         "the single-seed run.")
    ap.add_argument("--weight-split", default="forward_inner", choices=["forward_inner", "val"],
                    help="Surface to solve meta-weights on (forward_inner ensures no validation leakage)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    mlp_seeds = parse_seeds(args.mlp_seeds, args.seed)
    print(f"[{ARM}] MLP seeds averaged: {mlp_seeds}")

    print(f"[{ARM}] Loading base model predictions on weight solving split: {args.weight_split}...")
    
    Xtr, _, _ = D.load_part(args.split, "train", args.featureset, filter_target=True,
                            outlier_policy=args.outlier_policy)

    if args.weight_split == "forward_inner":
        wsplit = "forward_inner"
        Xw_tr, _, _ = D.load_part(wsplit, "train", args.featureset, filter_target=True,
                                  outlier_policy=args.outlier_policy)
        Xw, yw, meta_w = D.load_part(wsplit, "val", args.featureset, filter_target=False)
        yw = np.asarray(yw, dtype="float64")
        pa_w = predict_arm("per_airport", wsplit, "val", args.featureset, args.seed,
                           args.outlier_policy, X=Xw, meta=meta_w, Xtr=Xw_tr)
        pb_w = mlp_prediction(wsplit, "val", args.featureset, mlp_seeds,
                              args.outlier_policy, Xw, meta_w, Xw_tr)
        ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()
        weights_per_airport = solve_alpha_per_airport(yw, pa_w, pb_w, ap_w)
    else:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pa_val = predict_arm("per_airport", args.split, "val", args.featureset, args.seed,
                             args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
        pb_val = mlp_prediction(args.split, "val", args.featureset, mlp_seeds,
                                args.outlier_policy, Xva, meta_va, Xtr)
        ap_val = meta_va[AIRPORT_COL].astype("object").to_numpy()
        weights_per_airport = solve_alpha_per_airport(yva, pa_val, pb_val, ap_val)

    print(f"[{ARM}] Solved weights (weight on per_airport):")
    for k, v in weights_per_airport.items():
        print(f"  {k}: {v:.4f} (MLP weight: {1.0 - v:.4f})")

    # Evaluate validation RMSE on forward split
    has_val = bool(config.SPLITS[args.split]["val"])
    val_rmse = None
    if has_val:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pa_val = predict_arm("per_airport", args.split, "val", args.featureset, args.seed,
                             args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
        pb_val = mlp_prediction(args.split, "val", args.featureset, mlp_seeds,
                                args.outlier_policy, Xva, meta_va, Xtr)
        ap_val = meta_va[AIRPORT_COL].astype("object").to_numpy()
        p_val_blend = np.zeros(len(yva), dtype="float64")
        for a in pd.unique(ap_val):
            m = (ap_val == a)
            w = weights_per_airport.get(str(a), 0.5)
            p_val_blend[m] = w * pa_val[m] + (1.0 - w) * pb_val[m]
        p_val_blend = np.clip(p_val_blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, p_val_blend)
        print(f"[{ARM}] Forward Validation Blend RMSE: {val_rmse:.4f}")

    meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "weights_per_airport": weights_per_airport,
        "weight_split": args.weight_split,
        "mlp_seeds": mlp_seeds,
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        # compare.py's training-pool gate reads this key from every arm (the
        # blend arms per_airport_stack / ensemble_antigravity write it the same
        # way). It was missing here, so the gate would HALT the comparison rather
        # than rank this arm. Xtr is the forward train pool. MISTAKES.md #85.
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] Meta-model metadata saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
