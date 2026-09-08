#!/usr/bin/env python3
"""Arm tri_blend: ONE global NNLS blend over per_airport, per_operator, mlp_antigravity.

A different multi-model structure from the other blends in this build, tested
because the user asked for an alternative rather than only iterating on the
per-airport-weighted 2-arm blend (airport_operator_multimodel):

  - airport_operator_blend:        2 tree arms,  ONE global weight
  - multimodel_antigravity:        1 tree + MLP, PER-AIRPORT weight
  - airport_operator_multimodel:   1 tree-blend + MLP, PER-AIRPORT weight
  - tri_blend (this arm):          2 tree arms + MLP, ONE global weight vector

The per-airport blends already found the right mix varies by airport (LIRF
wants ~90% MLP, most others want ~95%+ tree); a single global weight vector
over three inputs cannot express that, so the honest prior is that this loses
to airport_operator_multimodel. It is still worth measuring directly rather
than assumed, and a 3-input global blend is a materially different free-
parameter count (3 weights, not up to 10 per-airport alphas) from the 9-cell
per-airport x-3-arm grid that overfit in-sample earlier this session
(246.94 in-sample vs 254.24 honest) -- NNLS with only 3 scalars fit on a
~180k-row inner holdout is a low-variance fit by comparison.

Weights are solved with non-negative least squares (scipy.optimize.nnls, same
solver as ensemble_antigravity) on forward_inner -- a holdout carved from
TRAIN, never val -- unlike ensemble_antigravity, which is a documented
exception in config.VAL_FITTED_ARMS that fits on val directly. Requires
per_airport, per_operator and mlp_antigravity to all be trained under the
forward_inner protocol for the target seed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)

ARM = "tri_blend"
BASE_ARMS = ("per_airport", "per_operator", "mlp_antigravity")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    Xtr, _, _ = D.load_part(args.split, "train", args.featureset, filter_target=True,
                            outlier_policy=args.outlier_policy)

    print(f"[{ARM}] solving weights on forward_inner (honest holdout carved from TRAIN)...")
    Xw_tr, _, _ = D.load_part("forward_inner", "train", args.featureset, filter_target=True,
                              outlier_policy=args.outlier_policy)
    Xw, yw, meta_w = D.load_part("forward_inner", "val", args.featureset, filter_target=False)
    yw = np.asarray(yw, dtype="float64")

    preds_w = []
    for a in BASE_ARMS:
        p = predict_arm(a, "forward_inner", "val", args.featureset, args.seed,
                        args.outlier_policy, X=Xw, meta=meta_w, Xtr=Xw_tr)
        preds_w.append(p)
        print(f"  base arm {a:16s} forward_inner RMSE: {rmse(yw, p):.3f}")

    P_w = np.column_stack(preds_w)
    raw_w, _ = nnls(P_w, yw)
    weights = {a: float(w) for a, w in zip(BASE_ARMS, raw_w)}
    print(f"[{ARM}] learned weights (unconstrained sum, NOT renormalised to 1):")
    for a, w in weights.items():
        print(f"  {a:16s}: {w:.4f}")

    inner_blend = np.clip(P_w @ raw_w, config.PREDICTION_CLIP_MIN_SEC,
                          config.PREDICTION_CLIP_MAX_SEC)
    inner_rmse_blend = rmse(yw, inner_blend)
    print(f"[{ARM}] forward_inner blend RMSE: {inner_rmse_blend:.3f}")

    # ---------------- evaluate on the real forward val split -------------------
    has_val = bool(config.SPLITS[args.split]["val"])
    val_rmse = None
    if has_val:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        preds_va = [predict_arm(a, args.split, "val", args.featureset, args.seed,
                                args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
                   for a in BASE_ARMS]
        P_va = np.column_stack(preds_va)
        p_val_blend = np.clip(P_va @ raw_w, config.PREDICTION_CLIP_MIN_SEC,
                              config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, p_val_blend)
        print(f"[{ARM}] Forward Validation Blend RMSE: {val_rmse:.4f}")
        for a, p in zip(BASE_ARMS, preds_va):
            print(f"  (for reference) {a:16s} alone val RMSE: {rmse(yva, np.clip(p, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)):.3f}")

    meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "base_arms": list(BASE_ARMS),
        "weights": weights,
        "weight_split": "forward_inner",
        "inner_rmse_blend": inner_rmse_blend,
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] Meta-model metadata saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
