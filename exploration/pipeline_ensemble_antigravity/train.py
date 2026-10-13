#!/usr/bin/env python3
"""Arm ensemble_antigravity: Constrained convex blend of candidate models.

Solves for non-negative weights summing to 1 minimizing validation RMSE across
available base models (lightgbm, ridge, naive_offblock, xgboost_antigravity).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.metrics import rmse
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)

ARM = "ensemble_antigravity"
POTENTIAL_BASE_ARMS = ["lightgbm", "xgboost_antigravity", "mlp_antigravity", "ridge", "naive_offblock"]


def predict_base_arm(arm: str, split: str, part: str, featureset: str, seed: int,
                     outlier_policy: str, X_part: pd.DataFrame,
                     Xtr: pd.DataFrame = None, tag: str = None) -> np.ndarray:
    # `tag` was missing here while common.predict.predict_arm resolved its own
    # run dir WITH the tag and then dropped it on the way in -- so
    # predict_arm(..., tag="congest") for any third-party arm silently loaded
    # the UNTAGGED run's weights and returned a different model's predictions.
    # Default None reproduces the previous path exactly for every caller.
    proto = config.protocol_key(split, featureset, outlier_policy)
    d = run_dir(config.ARTIFACTS / "runs", arm, protocol=proto, seed=seed,
                extra=({"tag": tag} if tag else None), create=False)
    if not d.exists():
        raise FileNotFoundError("Base arm '{}' run directory does not exist at {}".format(arm, d))

    if arm == "lightgbm":
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(d / "model.txt"))
        meta_json = json.loads((d / "model.json").read_text())
        X = D.align_columns(Xtr, X_part) if Xtr is not None else X_part
        pred = booster.predict(X, num_iteration=meta_json.get("best_iteration"))
    elif arm == "xgboost_antigravity":
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(d / "model.ubj"))
        meta_json = json.loads((d / "model.json").read_text())
        X = D.align_columns(Xtr, X_part) if Xtr is not None else X_part
        dmat = xgb.DMatrix(X, enable_categorical=True)
        best_iter = meta_json.get("best_iteration")
        iter_range = (0, best_iter + 1) if best_iter is not None else None
        pred = booster.predict(dmat, iteration_range=iter_range)
    elif arm == "mlp_antigravity":
        import torch
        from torch.utils.data import DataLoader
        from pipeline_mlp_antigravity.model import TabularMLP
        from pipeline_mlp_antigravity.train import TabularDataset, encode_data

        with open(d / "preprocessor.pkl", "rb") as fh:
            prep = pickle.load(fh)
        if prep.get("congest"):
            # The bundle records whether the net was fitted with the congestion
            # columns. encode_data indexes X_part by prep["num_cols"], so
            # without them this would either KeyError or -- if a caller had
            # already reindexed the frame -- feed five all-NaN columns through
            # fillna(medians) and silently score a different model.
            from pipeline_per_airport_congest.extra import attach as attach_congest
            X_part = attach_congest(X_part, "train2025", featureset)
        cat_tensors, num_tensor = encode_data(X_part, prep["cat_cols"], prep["num_cols"],
                                              prep["cat_maps"], prep["scaler"], prep["medians"])
        ds = TabularDataset(cat_tensors, num_tensor)
        loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        prior_mean = prep.get("prior_mean", 990.0)
        model = TabularMLP(prep["cat_cardinalities"], num_features=len(prep["num_cols"]),
                           hidden_dims=tuple(prep.get("hidden_dims", (256, 128, 64))),
                           dropout=prep.get("dropout", 0.1),
                           prior_mean=prior_mean,
                           residual=prep.get("residual", False)).to(device)
        model.load_state_dict(torch.load(d / "model.pt", map_location=device))
        model.eval()
        mlp_preds = []
        with torch.no_grad():
            for x_cat, x_num in loader:
                x_cat = {c: x_cat[c].to(device) for c in x_cat}
                x_num = x_num.to(device)
                mlp_preds.append(model(x_cat, x_num).cpu().numpy())
        pred = np.concatenate(mlp_preds)
    elif arm == "ridge":
        with open(d / "model.pkl", "rb") as fh:
            model = pickle.load(fh)
        pred = model.predict(X_part)
    elif arm == "naive_offblock":
        from pipeline_naive_offblock.evaluate import load_model
        from pipeline_naive_offblock.train import predict as naive_predict
        pred, _ = naive_predict(load_model(d), X_part, featureset)
    else:
        raise ValueError("Unknown base arm {!r}".format(arm))

    return np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--base-arms", default=None,
                    help="comma-separated list of base arms to blend; defaults to available")
    ap.add_argument("--freeze-from", default=None,
                    help="path to an honest run's model.json for refit")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    refit_only = not config.SPLITS[args.split]["val"]
    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                       filter_target=True,
                                       outlier_policy=args.outlier_policy)

    if refit_only:
        if not args.freeze_from:
            raise SystemExit("split '{}' has no validation set. Pass --freeze-from.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        weights = frozen["weights"]
        used_arms = frozen["base_arms"]
        val_rmse_clipped = None
    else:
        # Determine available base arms
        if args.base_arms:
            candidate_arms = [a.strip() for a in args.base_arms.split(",") if a.strip()]
        else:
            candidate_arms = []
            for a in POTENTIAL_BASE_ARMS:
                d = run_dir(config.ARTIFACTS / "runs", a, protocol=proto, seed=args.seed, create=False)
                if (d / "eval_val.json").exists() or (d / "model.json").exists() or (d / "model.pkl").exists() or (d / "tables.parquet").exists():
                    candidate_arms.append(a)

        if len(candidate_arms) < 2:
            raise RuntimeError(
                "Ensemble requires at least 2 base arms with trained models in {}, found: {}".format(
                    proto, candidate_arms))

        print("[{}] fitting ensemble over base arms: {}".format(ARM, candidate_arms))
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)

        preds = []
        used_arms = []
        for a in candidate_arms:
            try:
                p = predict_base_arm(a, args.split, "val", args.featureset, args.seed,
                                     args.outlier_policy, Xva, Xtr=Xtr)
                preds.append(p)
                used_arms.append(a)
                print("  base arm {:22s} val RMSE: {:.3f}".format(a, rmse(yva, p)))
            except Exception as e:
                print("  warning: failed to predict with base arm {}: {}".format(a, e))

        P = np.column_stack(preds)
        # Non-negative least squares
        raw_w, _ = nnls(P, np.asarray(yva, dtype="float64"))
        if raw_w.sum() > 0:
            norm_w = raw_w / raw_w.sum()
        else:
            norm_w = np.ones(len(used_arms)) / len(used_arms)

        weights = {arm: float(w) for arm, w in zip(used_arms, norm_w)}
        print("[{}] learned weights:".format(ARM))
        for arm, w in weights.items():
            print("  {:22s}: {:.4f}".format(arm, w))

        blend_val = np.clip(P @ norm_w, config.PREDICTION_CLIP_MIN_SEC,
                            config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, blend_val)
        print("[{}] blended val RMSE: {:.3f}".format(ARM, val_rmse_clipped))

    sel = declare_selection_set("ensemble_antigravity weights",
                                selected_on="frozen" if refit_only else "val",
                                value=weights)

    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "base_arms": used_arms, "weights": weights,
        "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "selection": dict(sel),
        "n_train": int(len(Xtr)),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(_pool_meta.attrs.get("outlier_policy", {})),
    }, force=args.force)

    print("[{}] saved model to {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
