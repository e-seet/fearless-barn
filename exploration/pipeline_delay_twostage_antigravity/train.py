#!/usr/bin/env python3
"""Training script for pipeline_delay_twostage_antigravity.

Trains a hurdle / two-stage model:
1. Hierarchical nominal baseline lookup b_i
2. Stage 1 Binary Classifier for delay occurrence: P(taxi_time - baseline > threshold)
3. Stage 2 Regressor for delay magnitude conditional on delay occurrence.
"""
from __future__ import annotations

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
from pipeline_delay_twostage_antigravity.model import TwoStageDelayModel

ARM = "delay_twostage_antigravity"

CLF_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
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

REG_PARAMS = {
    "objective": "l2",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 80,
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
    ap.add_argument("--delay-threshold", type=float, default=180.0,
                    help="Threshold in seconds above nominal baseline to classify as delayed")
    ap.add_argument("--num-boost-round", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=150)
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
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]

    model = TwoStageDelayModel(delay_threshold_sec=args.delay_threshold)
    model.fit_baselines(Xtr, ytr)

    b_tr = model.get_baselines(Xtr)
    delay_tr = np.maximum(0.0, ytr - b_tr)
    is_delayed_tr = (delay_tr > args.delay_threshold).astype(int)

    has_val = bool(config.SPLITS[args.split]["val"])
    val_set_clf = []
    val_set_reg = []
    Xva, yva, b_va, is_delayed_va, delay_va = None, None, None, None, None

    if has_val:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
        b_va = model.get_baselines(Xva)
        delay_va = np.maximum(0.0, yva - b_va)
        is_delayed_va = (delay_va > args.delay_threshold).astype(int)

        dva_clf = lgb.Dataset(Xva, label=is_delayed_va, categorical_feature=cats, free_raw_data=False)
        val_set_clf.append(dva_clf)

        # For regressor evaluation, evaluate on delayed subset
        delayed_va_idx = np.flatnonzero(is_delayed_va == 1)
        if len(delayed_va_idx) > 0:
            dva_reg = lgb.Dataset(Xva.iloc[delayed_va_idx], label=delay_va.iloc[delayed_va_idx] if hasattr(delay_va, "iloc") else delay_va[delayed_va_idx],
                                  categorical_feature=cats, free_raw_data=False)
            val_set_reg.append(dva_reg)

    print(f"[{ARM}] Fitting Classifier: P(Delay > {args.delay_threshold}s)... (Positive rate: {is_delayed_tr.mean():.3f})")
    dtr_clf = lgb.Dataset(Xtr, label=is_delayed_tr, categorical_feature=cats, free_raw_data=False)
    clf_callbacks = [lgb.early_stopping(args.patience, verbose=False)] if has_val else []
    clf_booster = lgb.train(
        CLF_PARAMS,
        dtr_clf,
        num_boost_round=args.num_boost_round,
        valid_sets=val_set_clf,
        valid_names=["val"] if has_val else None,
        callbacks=clf_callbacks,
    )
    model.clf = clf_booster

    print(f"[{ARM}] Fitting Regressor: E[Delay Magnitude | Delayed]...")
    delayed_tr_idx = np.flatnonzero(is_delayed_tr == 1)
    dtr_reg = lgb.Dataset(Xtr.iloc[delayed_tr_idx],
                          label=delay_tr.iloc[delayed_tr_idx] if hasattr(delay_tr, "iloc") else delay_tr[delayed_tr_idx],
                          categorical_feature=cats, free_raw_data=False)
    reg_callbacks = [lgb.early_stopping(args.patience, verbose=False)] if has_val else []
    reg_booster = lgb.train(
        REG_PARAMS,
        dtr_reg,
        num_boost_round=args.num_boost_round,
        valid_sets=val_set_reg,
        valid_names=["val"] if has_val else None,
        callbacks=reg_callbacks,
    )
    model.reg = reg_booster

    # Evaluate validation RMSE
    val_rmse = None
    if has_val:
        y_val_pred = model.predict(Xva)
        val_rmse = rmse(yva, y_val_pred)
        print(f"[{ARM}] Validation Combined RMSE: {val_rmse:.4f}")

    # Save model and metadata
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "delay_threshold_sec": args.delay_threshold,
        "clf_best_iteration": clf_booster.best_iteration,
        "reg_best_iteration": reg_booster.best_iteration,
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        # compare.py's training-pool gate reads this exact key (as every
        # pipeline_* arm writes it). It was named `train_pool_fingerprint` here,
        # so the gate saw no hash and would have HALTed the comparison. Same
        # value, correct key. See MISTAKES.md #85.
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] Trained run saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
