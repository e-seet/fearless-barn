#!/usr/bin/env python3
"""Arm airport_wtc_rec_multimodel: airport_operator_multimodel with a refined
3-way operational segmentation per airport (Wake Turbulence Category + Flight Record status).

Why Wake Turbulence Category + Flight Record status
---------------------------------------------------
airport_wtc_multimodel established that aircraft size/class (WTC Heavy+ vs Other)
is a real, physical axis for determining the optimal tree/neural blend weight:
freighters and wide-body aircraft at LIRF strongly prefer the tree model (alpha ~0.83),
whereas non-heavy aircraft prefer the MLP (alpha ~0.08).

However, data audit and target analysis in config.py / stage1_audit revealed another
crucial physical subpopulation:
82.4% of extreme taxi-out records (>7,200s) have no joined flight record
(has_flight_record == 0), representing uncorrelated, unscheduled, or anomalous
flight operations. In airport_wtc_multimodel, these extreme records fall into the
'Other' bucket and exert massive squared-error gradient pull on L2 optimization,
forcing normal scheduled passenger flights to take a near-pure neural blend (alpha 0.08)
even though the tree model performs substantially better on normal traffic.

This arm partitions each airport into three operational regimes:
1. 'Heavy+'    : Wake turbulence categories H and J (wide-body and freighters).
2. 'NoRecord'  : Movements without joined flight records (has_flight_record == 0).
3. 'Other'     : Standard scheduled commercial passenger operations.

Every cell across all 10 airports contains at least 308 rows on forward_inner,
comfortably exceeding the minimum threshold required to reliably solve a 1D scalar
alpha without overfitting.

What this arm tests
-------------------
Reuses the trained airport_operator_blend and mlp_antigravity base models and solves
the least-squares blend weights out-of-sample on forward_inner for each
(airport, operational_bucket) cell.
"""
from __future__ import annotations

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

ARM = "airport_wtc_rec_multimodel"
WTC_COL = "WK_TBL_CAT_flt"
REC_COL = "has_flight_record"
DEFAULT_MIN_CELL_ROWS = 300


def _bucket_key(ap, X) -> np.ndarray:
    wtc = X[WTC_COL].astype("string").fillna("NaN").to_numpy()
    has_rec = X[REC_COL].to_numpy()
    
    is_heavy = np.isin(wtc, ["H", "J"])
    is_norec = (has_rec == 0)
    
    bucket = np.where(is_heavy, "Heavy+", np.where(is_norec, "NoRecord", "Other"))
    return np.array(["{}|{}".format(a, b) for a, b in zip(ap, bucket)])


def solve_alpha_per_cell(y, pa, pb, ap, X, min_cell_rows) -> tuple[dict, dict, list]:
    """One alpha per (airport, operational bucket); falls back to the airport-level
    alpha (all buckets pooled) for any cell below min_cell_rows."""
    cell = _bucket_key(ap, X)
    airport_alpha = {}
    for a in sorted(pd.unique(ap)):
        m = (ap == a)
        d = pa[m] - pb[m]
        r = y[m] - pb[m]
        denom = float(np.dot(d, d))
        airport_alpha[str(a)] = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))

    cell_alpha = {}
    fallback_cells = []
    for c in sorted(pd.unique(cell)):
        m = (cell == c)
        n = int(m.sum())
        a = c.split("|", 1)[0]
        if n < min_cell_rows:
            cell_alpha[c] = airport_alpha[a]
            fallback_cells.append(c)
            continue
        d = pa[m] - pb[m]
        r = y[m] - pb[m]
        denom = float(np.dot(d, d))
        cell_alpha[c] = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))
    return cell_alpha, airport_alpha, fallback_cells


def blend_with_cell_alpha(pa, pb, ap, X, cell_alpha, airport_alpha) -> np.ndarray:
    cell = _bucket_key(ap, X)
    out = np.zeros(len(pa), dtype="float64")
    for c in pd.unique(cell):
        m = (cell == c)
        a = c.split("|", 1)[0]
        w = cell_alpha.get(c, airport_alpha.get(a, 0.5))
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
    ap_.add_argument("--min-cell-rows", type=int, default=DEFAULT_MIN_CELL_ROWS)
    ap_.add_argument("--force", action="store_true")
    args = ap_.parse_args(argv)

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
    pa_w = predict_arm("airport_operator_blend", "forward_inner", "val", args.featureset, args.seed,
                       args.outlier_policy, X=Xw, meta=meta_w, Xtr=Xw_tr)
    pb_w = predict_arm("mlp_antigravity", "forward_inner", "val", args.featureset, args.seed,
                       args.outlier_policy, X=Xw, meta=meta_w, Xtr=Xw_tr)
    ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()

    cell_alpha, airport_alpha, fallback_cells = solve_alpha_per_cell(
        yw, pa_w, pb_w, ap_w, Xw, args.min_cell_rows)

    print(f"[{ARM}] solved cell weights (weight on airport_operator_blend):")
    for k, v in sorted(cell_alpha.items()):
        flag = " (fallback to airport alpha)" if k in fallback_cells else ""
        print(f"  {k:25s}: {v:.4f}{flag}")

    inner_blend = blend_with_cell_alpha(pa_w, pb_w, ap_w, Xw, cell_alpha, airport_alpha)
    inner_blend = np.clip(inner_blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    print(f"[{ARM}] forward_inner blend RMSE: {rmse(yw, inner_blend):.3f}")

    has_val = bool(config.SPLITS[args.split]["val"])
    val_rmse = None
    if has_val:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pa_val = predict_arm("airport_operator_blend", args.split, "val", args.featureset, args.seed,
                             args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
        pb_val = predict_arm("mlp_antigravity", args.split, "val", args.featureset, args.seed,
                             args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
        ap_val = meta_va[AIRPORT_COL].astype("object").to_numpy()
        p_val_blend = blend_with_cell_alpha(pa_val, pb_val, ap_val, Xva, cell_alpha, airport_alpha)
        p_val_blend = np.clip(p_val_blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, p_val_blend)
        print(f"[{ARM}] Forward Validation Blend RMSE: {val_rmse:.4f}")

    meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "wtc_col": WTC_COL,
        "rec_col": REC_COL,
        "cell_alpha": cell_alpha,
        "airport_alpha_fallback": airport_alpha,
        "fallback_cells": fallback_cells,
        "min_cell_rows": args.min_cell_rows,
        "weight_split": "forward_inner",
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)
    print(f"[{ARM}] Meta-model metadata saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
