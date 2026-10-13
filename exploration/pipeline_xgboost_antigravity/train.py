#!/usr/bin/env python3
"""Arm xgboost_antigravity: XGBoost with histogram tree method and native categoricals.

Why XGBoost here:
Provides tree-based diversity against LightGBM on 1.4M rows x ~50 features.
Uses histogram-based split finding (`tree_method='hist'`) with `enable_categorical=True`
to natively split on high-cardinality categoricals (stand_runway, adep_stand, etc.).
Objective is `reg:squarederror` (L2 loss matching the leaderboard metric).
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
from common import features as F
from common.metrics import rmse
from common.bootstrap import (convergence_evidence, declare_selection_set, run_dir,
                              seed_everything, write_json_guarded)

ARM = "xgboost_antigravity"

PARAMS = {
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "grow_policy": "lossguide",
    "max_leaves": 255,
    "max_depth": 0,
    "min_child_weight": 100.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000,
                    help="SAFETY CAP, not the stopping condition (see --patience)")
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--max-leaves", type=int, default=PARAMS["max_leaves"])
    ap.add_argument("--max-depth", type=int, default=PARAMS["max_depth"])
    ap.add_argument("--min-child-weight", type=float, default=PARAMS["min_child_weight"])
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "gpu"],
                    help="XGBoost device ('cpu' or 'cuda')")
    ap.add_argument("--freeze-from", default=None,
                    help="path to an honest run's model.json for no-val 'full' refit")
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES),
                    help="how the TRAINING tail is handled; part of the artifact key")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import xgboost as xgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                       filter_target=True,
                                       outlier_policy=args.outlier_policy)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set, so nothing can be selected on "
                "it. Pass --freeze-from <honest run>/model.json to supply the "
                "hyperparameters and the round count.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit(
                "--freeze-from points at a run on split '{}', which is not a valid "
                "selection surface. Freeze from a 'forward' run.".format(frozen["split"]))
        print("[{}] REFIT on '{}' with hyperparameters frozen from {} "
              "(split={}, featureset={}, best_iteration={})".format(
                  ARM, args.split, args.freeze_from, frozen["split"],
                  frozen["featureset"], frozen["best_iteration"]))
    else:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)

    print("[{}] train n={} val n={} feats={} cats={} device={}".format(
        ARM, len(Xtr), 0 if refit_only else len(Xva), Xtr.shape[1], len(cats),
        args.device))

    device_str = "cuda" if args.device in ("cuda", "gpu") else "cpu"

    if frozen:
        params = dict(frozen["params"])
        params.update({"seed": args.seed, "device": device_str})
    else:
        params = dict(PARAMS)
        params.update({
            "learning_rate": args.learning_rate,
            "max_leaves": args.max_leaves,
            "max_depth": args.max_depth,
            "min_child_weight": args.min_child_weight,
            "seed": args.seed,
            "device": device_str,
        })

    dtr = xgb.DMatrix(Xtr, label=np.asarray(ytr, dtype="float64"), enable_categorical=True)

    if refit_only:
        ratio = len(Xtr) / max(1, frozen["n_train"])
        n_rounds = max(1, int(round(frozen["best_iteration"] * ratio)))
        print("  frozen best_iteration {} x pool ratio {:.3f} -> {} rounds".format(
            frozen["best_iteration"], ratio, n_rounds))
        booster = xgb.train(params, dtr, num_boost_round=n_rounds, verbose_eval=100)
        best_iteration = n_rounds - 1
        conv = {"refit_only": True, "hit_max_epochs": False,
                "rounds": n_rounds, "frozen_from": args.freeze_from,
                "frozen_best_iteration": frozen["best_iteration"],
                "pool_ratio": ratio,
                "note": "no early stopping: round count is frozen, not selected"}
        val_curve = []
    else:
        dva = xgb.DMatrix(Xva, label=np.asarray(yva, dtype="float64"), enable_categorical=True)
        evals_result = {}
        booster = xgb.train(
            params,
            dtr,
            num_boost_round=args.num_boost_round,
            evals=[(dtr, "train"), (dva, "val")],
            evals_result=evals_result,
            early_stopping_rounds=args.patience,
            verbose_eval=100,
        )
        best_iteration = int(booster.best_iteration)
        val_curve = evals_result["val"]["rmse"]
        hist = [{"epoch": i + 1, "val_rmse": v} for i, v in enumerate(val_curve)]
        conv = convergence_evidence(hist, max_epochs=args.num_boost_round,
                                    patience=args.patience, metric_key="val_rmse",
                                    mode="min")
        print("  best_iteration={} of cap {} | early stopping fired: {}".format(
            best_iteration, args.num_boost_round,
            not conv.get("hit_max_epochs", True)))
        if conv.get("hit_max_epochs"):
            print("  WARNING: hit the round cap -- this is a signal to raise the cap.")

    sel = declare_selection_set(
        "xgboost_antigravity best_iteration",
        selected_on="frozen" if refit_only else "val",
        value=best_iteration)

    booster.save_model(str(out / "model.ubj"))

    # Feature importance
    scores = booster.get_score(importance_type="gain")
    imp_rows = [{"feature": col, "gain": float(scores.get(col, 0.0))} for col in Xtr.columns]
    imp = pd.DataFrame(imp_rows)
    imp.sort_values("gain", ascending=False).to_parquet(out / "feature_importance.parquet", index=False)

    if refit_only:
        val_rmse_raw = val_rmse_clipped = None
    else:
        val_pred = booster.predict(dva, iteration_range=(0, best_iteration + 1))
        val_pred = np.clip(val_pred, config.PREDICTION_CLIP_MIN_SEC,
                           config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_raw = float(val_curve[best_iteration])
        val_rmse_clipped = rmse(yva, val_pred)

    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration": best_iteration,
        "val_rmse_raw": val_rmse_raw, "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "convergence": conv, "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(_pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)

    print("[{}] {} -> {}".format(
        ARM,
        "refit complete (no val by design)" if refit_only
        else "val RMSE (clipped) = {:.3f}".format(val_rmse_clipped), out))
    print("  top-15 features by gain:")
    for r in imp.sort_values("gain", ascending=False).head(15).itertuples():
        print("    {:28s} gain={:14.0f}".format(r.feature, r.gain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
