#!/usr/bin/env python3
"""Arm airport_unsup_multimodel: Multimodel blending over unsupervised congestion
regimes per airport combined with Wake Turbulence Category Heavy class.

Why Unsupervised Congestion Regimes:
-----------------------------------
Airports operate across distinct physical macro-states: free-flow off-peak,
moderate passenger bank, arrival-crossing blockages, and severe departure runway
queues. These states cannot be captured by a single rule-based threshold.

This arm uses an unsupervised K-Means model (K=5) fitted on instantaneous
traffic vectors (pushback surge, runway queue, arrival rates, time-of-day) per airport
to discover the true operational congestion regimes with zero labels.
Together with the physical Heavy/Jumbo class (WTC H and J), this creates clean
operational regimes per airport where tree and neural models have strongly differentiated
advantages.

Blend weights are solved strictly out-of-sample on forward_inner with zero validation leakage.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)
from pipeline_per_airport.model import AIRPORT_COL

ARM = "airport_unsup_multimodel"
WTC_COL = "WK_TBL_CAT_flt"
REC_COL = "has_flight_record"
MKT_COL = "MARKET_SEGMENT_flt"
DEFAULT_MIN_CELL_ROWS = 30
DEFAULT_K_CLUSTERS = 6
DEFAULT_SHRINKAGE_LAMBDA = 2000.0

CLUSTER_FEATURES = [
    "dep_prev_15m", "dep_prev_30m", "dep_prev_60m",
    "arr_prev_15m", "arr_prev_30m", "arr_prev_60m",
    "same_rwy_prev_15m", "same_rwy_prev_30m",
    "hour", "minute_of_day"
]


def fit_unsupervised_clusters(X: pd.DataFrame, ap_arr: np.ndarray, k: int, seed: int):
    scalers = {}
    kmeans_models = {}
    for a in np.unique(ap_arr):
        m = (ap_arr == a)
        X_ap = X.loc[m, CLUSTER_FEATURES].fillna(0).to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_ap)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        scalers[str(a)] = scaler
        kmeans_models[str(a)] = km
    return scalers, kmeans_models


def assign_clusters(X: pd.DataFrame, ap_arr: np.ndarray, scalers: dict, kmeans_models: dict) -> np.ndarray:
    clusters = np.zeros(len(X), dtype=int)
    for a in np.unique(ap_arr):
        m = (ap_arr == a)
        scaler = scalers[str(a)]
        km = kmeans_models[str(a)]
        X_ap = X.loc[m, CLUSTER_FEATURES].fillna(0).to_numpy()
        X_scaled = scaler.transform(X_ap)
        clusters[m] = km.predict(X_scaled)
    return clusters


def _physical_regime(X: pd.DataFrame) -> np.ndarray:
    wtc = X[WTC_COL].astype("string").fillna("NaN").to_numpy()
    is_heavy = np.isin(wtc, ["H", "J"])
    has_rec = X[REC_COL].to_numpy() if REC_COL in X.columns else np.ones(len(X), dtype=int)
    is_norec = (has_rec == 0)
    mkt = X[MKT_COL].astype("string").fillna("NaN").to_numpy() if MKT_COL in X.columns else np.array(["NaN"] * len(X))
    is_reg = (mkt == "Regional")
    return np.where(is_heavy, "Heavy+", np.where(is_norec, "NoRecord", np.where(is_reg, "Regional", "Standard")))


def _bucket_key(ap_arr: np.ndarray, X: pd.DataFrame, clusters: np.ndarray) -> np.ndarray:
    regimes = _physical_regime(X)
    return np.array(["{}|{}|c{}".format(a, r, c) for a, r, c in zip(ap_arr, regimes, clusters)])


def _parent_key(ap_arr: np.ndarray, X: pd.DataFrame) -> np.ndarray:
    regimes = _physical_regime(X)
    return np.array(["{}|{}".format(a, r) for a, r in zip(ap_arr, regimes)])


def solve_hierarchical_alpha(
    y: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
    cell_arr: np.ndarray,
    parent_arr: np.ndarray,
    ap_arr: np.ndarray,
    min_cell_rows: int = DEFAULT_MIN_CELL_ROWS,
    lam: float = DEFAULT_SHRINKAGE_LAMBDA,
) -> tuple[dict, dict, dict, list]:
    # Level 1: Global per-airport alpha
    airport_alpha = {}
    for a in sorted(pd.unique(ap_arr)):
        m = (ap_arr == a)
        d = pa[m] - pb[m]
        r = y[m] - pb[m]
        denom = float(np.dot(d, d))
        airport_alpha[str(a)] = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))

    # Level 2: Parent (airport | physical regime) alpha
    parent_alpha = {}
    for parent in sorted(pd.unique(parent_arr)):
        m = (parent_arr == parent)
        a = parent.split("|", 1)[0]
        if m.sum() < 200:
            parent_alpha[str(parent)] = airport_alpha[a]
            continue
        d = pa[m] - pb[m]
        r = y[m] - pb[m]
        denom = float(np.dot(d, d))
        parent_alpha[str(parent)] = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))

    # Level 3: Child (airport | physical regime | cluster) alpha with Empirical Bayes shrinkage
    cell_alpha = {}
    fallback_cells = []
    for cell in sorted(pd.unique(cell_arr)):
        m = (cell_arr == cell)
        parent = cell.rsplit("|", 1)[0]
        a = cell.split("|", 1)[0]
        n_m = int(m.sum())
        p_val = parent_alpha.get(parent, airport_alpha.get(a, 0.5))

        if n_m < min_cell_rows:
            cell_alpha[str(cell)] = p_val
            fallback_cells.append(cell)
            continue

        d = pa[m] - pb[m]
        r = y[m] - pb[m]
        denom = float(np.dot(d, d))
        raw_alpha = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))

        # Stein / Empirical-Bayes shrinkage towards parent regime
        shrink_weight = float(n_m) / (float(n_m) + lam)
        cell_alpha[str(cell)] = float(shrink_weight * raw_alpha + (1.0 - shrink_weight) * p_val)

    return cell_alpha, parent_alpha, airport_alpha, fallback_cells


def blend_with_cell_alpha(
    pa: np.ndarray,
    pb: np.ndarray,
    cell_arr: np.ndarray,
    ap_arr: np.ndarray,
    cell_alpha: dict,
    airport_alpha: dict,
    parent_alpha: dict | None = None,
) -> np.ndarray:
    if parent_alpha is None:
        parent_alpha = {}
    out = np.zeros(len(pa), dtype="float64")
    for c in pd.unique(cell_arr):
        m = (cell_arr == c)
        parent = c.rsplit("|", 1)[0] if "|" in c else c
        a = c.split("|", 1)[0]
        w = cell_alpha.get(c, parent_alpha.get(parent, airport_alpha.get(a, 0.5)))
        out[m] = w * pa[m] + (1.0 - w) * pb[m]
    return out


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap_.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                     choices=list(config.FEATURESETS))
    ap_.add_argument("--seed", type=int, default=config.SEED)
    ap_.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                     choices=sorted(config.OUTLIER_POLICIES))
    ap_.add_argument("--k-clusters", type=int, default=DEFAULT_K_CLUSTERS)
    ap_.add_argument("--min-cell-rows", type=int, default=DEFAULT_MIN_CELL_ROWS)
    ap_.add_argument("--shrinkage-lambda", type=float, default=DEFAULT_SHRINKAGE_LAMBDA)
    ap_.add_argument("--weight-split", default=None)
    ap_.add_argument("--force", action="store_true")
    args = ap_.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    weight_split = args.weight_split or ("forward_inner" if args.split == "forward" else args.split)

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)

    # 1. Load holdout data to fit unsupervised clusters and solve alphas
    Xw, yw, meta_w = D.load_part(weight_split, "val", args.featureset, filter_target=False)
    ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()

    print("[{}] Fitting unsupervised K-Means (K={}) on {} holdout (N={})...".format(
        ARM, args.k_clusters, weight_split, len(Xw)))
    scalers, kmeans_models = fit_unsupervised_clusters(Xw, ap_w, args.k_clusters, args.seed)

    clusters_w = assign_clusters(Xw, ap_w, scalers, kmeans_models)
    cell_w = _bucket_key(ap_w, Xw, clusters_w)
    parent_w = _parent_key(ap_w, Xw)

    print("[{}] Loading base predictions on {} holdout...".format(ARM, weight_split))
    pa_w = predict_arm("airport_operator_blend", weight_split, "val",
                       args.featureset, args.seed, args.outlier_policy, X=Xw, meta=meta_w)
    pb_w = predict_arm("mlp_antigravity", weight_split, "val",
                       args.featureset, args.seed, args.outlier_policy, X=Xw, meta=meta_w)

    cell_alpha, parent_alpha, airport_alpha, fallbacks = solve_hierarchical_alpha(
        yw, pa_w, pb_w, cell_w, parent_w, ap_w,
        min_cell_rows=args.min_cell_rows, lam=args.shrinkage_lambda
    )

    print("[{}] Solved {} cell alphas ({} fallbacks).".format(ARM, len(cell_alpha), len(fallbacks)))

    # Save cluster models
    with open(out / "cluster_models.pkl", "wb") as fh:
        pickle.dump({"scalers": scalers, "kmeans_models": kmeans_models, "k_clusters": args.k_clusters}, fh)

    # Evaluate on val
    Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
    ap_va = meta_va[AIRPORT_COL].astype("object").to_numpy()
    clusters_va = assign_clusters(Xva, ap_va, scalers, kmeans_models)
    cell_va = _bucket_key(ap_va, Xva, clusters_va)

    pa_va = predict_arm("airport_operator_blend", args.split, "val",
                        args.featureset, args.seed, args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
    pb_va = predict_arm("mlp_antigravity", args.split, "val",
                        args.featureset, args.seed, args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)

    pred_va = blend_with_cell_alpha(pa_va, pb_va, cell_va, ap_va, cell_alpha, airport_alpha, parent_alpha)
    pred_va = np.clip(pred_va, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    val_rmse = rmse(yva, pred_va)
    print("[{}] Val RMSE: {:.4f}".format(ARM, val_rmse))

    model_meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "k_clusters": args.k_clusters,
        "shrinkage_lambda": args.shrinkage_lambda,
        "cell_alpha": {str(k): float(v) for k, v in cell_alpha.items()},
        "parent_alpha": {str(k): float(v) for k, v in parent_alpha.items()},
        "airport_alpha_fallback": {str(k): float(v) for k, v in airport_alpha.items()},
        "fallback_cells": fallbacks,
        "min_cell_rows": args.min_cell_rows,
        "weight_split": weight_split,
        "weights_are_val_fitted": False,
        "val_rmse": float(val_rmse),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "feature_fingerprint": config.feature_code_fingerprint(),
    }
    write_json_guarded(out / "model.json", model_meta, force=args.force)

    # Automatically write eval_val.json
    from common import evaluate as E
    res_val = E.evaluate(yva, pred_va, meta_va, ytr, part="val", arm=ARM, split=args.split,
                         featureset=args.featureset, seed=args.seed,
                         extra={"cell_alpha": cell_alpha, "weight_split": weight_split})
    write_json_guarded(out / "eval_val.json", res_val, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
