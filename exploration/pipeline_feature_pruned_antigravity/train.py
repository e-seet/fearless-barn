#!/usr/bin/env python3
"""Training script for pipeline_feature_pruned_antigravity.

Applies systematic feature pruning:
- Removes out-of-range calendar extrapolation artifacts (month_num, doy, doy_sin, doy_cos).
- Evaluates LightGBM with clean causal congestion + local stand/runway geometry.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import features as F
from common.metrics import rmse
from common.bootstrap import (convergence_evidence, declare_selection_set, run_dir,
                              seed_everything, write_json_guarded)

ARM = "feature_pruned_antigravity"

# Explicitly prune noisy out-of-distribution temporal predictors
DROP_FEATURES = ["month_num", "doy", "doy_sin", "doy_cos"]

PARAMS = {
    "objective": "l2",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "verbosity": -1,
    "num_threads": 0,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                       filter_target=True,
                                       outlier_policy=args.outlier_policy)
    
    # Prune target features
    Xtr = Xtr.drop(columns=[c for c in DROP_FEATURES if c in Xtr.columns])
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]

    has_val = bool(config.SPLITS[args.split]["val"])
    valid_sets = None
    valid_names = None
    Xva, yva = None, None

    if has_val:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
        Xva = Xva.drop(columns=[c for c in DROP_FEATURES if c in Xva.columns])
        dva = lgb.Dataset(Xva, label=yva, categorical_feature=cats, free_raw_data=False)
        valid_sets = [dva]
        valid_names = ["val"]

    dtr = lgb.Dataset(Xtr, label=ytr, categorical_feature=cats, free_raw_data=False)
    callbacks = [lgb.early_stopping(args.patience, verbose=False)] if has_val else []

    print(f"[{ARM}] Training LightGBM on {Xtr.shape[1]} pruned features (pruned: {DROP_FEATURES})...")
    booster = lgb.train(
        PARAMS,
        dtr,
        num_boost_round=args.num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    val_rmse = None
    if has_val:
        preds = np.clip(booster.predict(Xva), config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, preds)
        print(f"[{ARM}] Val RMSE: {val_rmse:.4f}")

    booster.save_model(str(out / "model.txt"))
    meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "best_iteration": int(booster.best_iteration or args.num_boost_round),
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        "pruned_features": DROP_FEATURES,
        # compare.py's training-pool gate reads this exact key; it was named
        # `train_pool_fingerprint` here and the gate saw no hash. MISTAKES.md #85.
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] Model saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
