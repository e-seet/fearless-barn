#!/usr/bin/env python3
"""Arm airport_wtc_multimodel: airport_operator_multimodel with a finer blend
weight -- per (airport, wake-turbulence-category bucket) instead of per airport.

Why wake-turbulence category, not runway or a raw "flight type" split
-----------------------------------------------------------------------
The user asked to chain runway and "cargo type" into the multi-model blend.
Two things worth being explicit about instead of guessing:

1. Runway identity is ALREADY a feature every arm in this build uses directly
   -- RUNWAY_mvt, adep_runway and stand_runway are native categoricals in the
   base feature set, and the SHAP dependence check earlier this session found
   adep_runway is one of the largest single contributors at LIRF (137-163s).
   So "does runway matter" is already answered yes, and every tree-based arm
   here already exploits it. What would be NEW is routing by runway, and that
   was checked before building anything: RUNWAY_mvt's per-airport cardinality
   is highly skewed (checked empirically -- some airports have a handful of
   runway codes with near-zero counts, e.g. LIRF and EGLL show ZERO rows for
   the 'L' wake category and several runway codes barely register). Nesting a
   trained booster per (airport, runway) would very likely reproduce the
   per_airport_flighttype failure (see its docstring/catalog entry: LIRF's
   thin 'Other' cell wrecked that arm, val 295.77 vs 278.17) -- worse, since
   runway has more categories than flight-type did. Not built for that reason;
   flagged here rather than silently skipped.

2. There is no direct "cargo" flag in this data. The honest real-data proxy
   for "can this aircraft carry a lot" is WK_TBL_CAT_flt (ICAO wake turbulence
   category: L/M/H/J), already a categorical feature in the base set. 'H'
   (heavy) and 'J' (super, e.g. A380) are the categories that include
   freighters and the largest wide-bodies. Checked empirically before
   building: bucketing to Heavy+ (H/J) vs Other (M/L/NaN) keeps every one of
   the 20 (airport, bucket) cells above ~2,600 rows on the forward_inner
   weight-solving holdout -- comfortably enough to fit ONE scalar alpha per
   cell (this is a least-squares weight fit, not a trained booster, so the
   data bar is far lower than per_airport_flighttype's failure mode).

What this arm actually tests
-----------------------------
Reuses the already-trained airport_operator_blend and mlp_antigravity (no new
boosters), and re-solves multimodel's per-airport alpha at (airport, WTC
bucket) granularity instead: does the best tree/neural mix for heavy/super
aircraft differ from the mix for everything else at the same airport? A cell
falling below --min-cell-rows falls back to that airport's single alpha
(computed the same way as airport_operator_multimodel), so a thin cell cannot
silently degrade the fit.
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

ARM = "airport_wtc_multimodel"
WTC_COL = "WK_TBL_CAT_flt"
DEFAULT_MIN_CELL_ROWS = 1000


def _bucket_key(ap, X) -> np.ndarray:
    wtc = X[WTC_COL].astype("string").fillna("NaN")
    heavy = wtc.isin(["H", "J"]).to_numpy()
    bucket = np.where(heavy, "Heavy+", "Other")
    return np.array(["{}|{}".format(a, b) for a, b in zip(ap, bucket)])


def solve_alpha_per_cell(y, pa, pb, ap, X, min_cell_rows) -> dict:
    """One alpha per (airport, WTC bucket); falls back to the airport-level
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
        print(f"  {k:20s}: {v:.4f}{flag}")

    inner_blend = blend_with_cell_alpha(pa_w, pb_w, ap_w, Xw, cell_alpha, airport_alpha)
    inner_blend = np.clip(inner_blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    print(f"[{ARM}] forward_inner blend RMSE: {rmse(yw, inner_blend):.3f}  "
          f"(airport_operator_multimodel-equivalent, single-alpha-per-airport, "
          f"would give: {rmse(yw, blend_with_cell_alpha(pa_w, pb_w, ap_w, Xw, {}, airport_alpha)):.3f})")

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
