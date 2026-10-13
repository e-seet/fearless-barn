#!/usr/bin/env python3
"""Training and meta-weight solving for pipeline_airport_operator_multimodel.

Solves per-airport convex blend weights combining:
1. airport_operator_blend (global blend of per_airport + per_operator; val 272.88,
   the best tree-only arm in this build)
2. mlp_antigravity (Tabular neural net with entity embeddings; val 253.99, the
   best single arm overall)

Same lever as multimodel_antigravity (per_airport + mlp_antigravity, val 248.81)
and cross_blend, with a different tree-side parent. multimodel_antigravity wins
on val by ~24s over every tree-only arm built this session because per-airport
weighting lets the blend lean on the MLP exactly where the tree family is worst
(e.g. LIRF). airport_operator_blend is strictly better than per_airport alone,
so substituting it in is the direct, cheap test of whether that improvement
carries through the neural blend too.

Weights are solved on forward_inner (a holdout carved from TRAIN, never val),
same discipline as multimodel_antigravity -- see its docstring and MISTAKES.md's
lever-3 lesson (in-sample blend weights measured 246.94, honest out-of-sample
254.24/252.68 -- WORSE than baseline). This arm additionally requires
airport_operator_blend to itself be trained under the forward_inner protocol
(fit on forward_inner's train pool, which already excludes the held-out month)
so that predict_arm("airport_operator_blend", "forward_inner", "val", ...) is
not leaking predictions from a model that saw the held-out month during its own
training. Run pipeline_airport_operator_blend.train --split forward_inner first.
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

ARM = "airport_operator_multimodel"


def parse_seeds(spec: str, default_seed: int) -> "list[int]":
    """'' -> [default_seed]; '42,43,44' -> [42, 43, 44]."""
    if not spec:
        return [int(default_seed)]
    return [int(x) for x in str(spec).split(",") if str(x).strip() != ""]


def mlp_prediction(split, part, featureset, seeds, outlier_policy, X, meta, Xtr,
                   tag=None):
    """mlp_antigravity's prediction, averaged over `seeds`.

    A single seed reproduces the original behaviour exactly (mean of one).

    Why average the MLP and not the whole arm: the tree side of this blend is
    almost deterministic across seeds (val RMSE SD 0.27 for per_airport, 0.80
    for airport_operator_blend) while mlp_antigravity's is 3.52 -- so
    essentially all of this blend's seed variance enters through the MLP.
    Averaging independent draws of the noisy component is variance reduction
    with ZERO fitted parameters.

    This is deliberately NOT what airport_multiseed_multimodel did. That arm
    averaged the final predictions of an over-tuned base and so averaged its
    optimism; this averages a noisy PARENT underneath an honest blend, which is
    the distinction the pipeline review drew explicitly.
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
    ap.add_argument("--mlp-tag", default=None,
                    help="Tag of the mlp_antigravity runs to average (e.g. "
                         "congest). Applies to every seed in --mlp-seeds. "
                         "Without this the neural side can only ever be the "
                         "untagged variant.")
    ap.add_argument("--tree-tag", default=None,
                    help="Tag of the airport_operator_blend run to use as the "
                         "tree-side parent (e.g. congest). Without this the "
                         "blend can only ever be built on the untagged parent.")
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
        pa_w = predict_arm("airport_operator_blend", wsplit, "val", args.featureset, args.seed,
                           args.outlier_policy, X=Xw, meta=meta_w, Xtr=Xw_tr,
                           tag=args.tree_tag)
        pb_w = mlp_prediction(wsplit, "val", args.featureset, mlp_seeds,
                              args.outlier_policy, Xw, meta_w, Xw_tr,
                              tag=args.mlp_tag)
        ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()
        weights_per_airport = solve_alpha_per_airport(yw, pa_w, pb_w, ap_w)
    else:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pa_val = predict_arm("airport_operator_blend", args.split, "val", args.featureset, args.seed,
                             args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr,
                             tag=args.tree_tag)
        pb_val = mlp_prediction(args.split, "val", args.featureset, mlp_seeds,
                                args.outlier_policy, Xva, meta_va, Xtr,
                                tag=args.mlp_tag)
        ap_val = meta_va[AIRPORT_COL].astype("object").to_numpy()
        weights_per_airport = solve_alpha_per_airport(yva, pa_val, pb_val, ap_val)

    print(f"[{ARM}] Solved weights (weight on airport_operator_blend):")
    for k, v in weights_per_airport.items():
        print(f"  {k}: {v:.4f} (MLP weight: {1.0 - v:.4f})")

    # Evaluate validation RMSE on forward split
    has_val = bool(config.SPLITS[args.split]["val"])
    val_rmse = None
    if has_val:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pa_val = predict_arm("airport_operator_blend", args.split, "val", args.featureset, args.seed,
                             args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr,
                             tag=args.tree_tag)
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
        "tree_tag": args.tree_tag,
        "mlp_tag": args.mlp_tag,
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] Meta-model metadata saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
