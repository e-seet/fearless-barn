#!/usr/bin/env python3
"""Arm airport_runway_multimodel: airport_operator_multimodel with the blend
weight solved per (airport, primary-runway-vs-other) instead of per airport.

Same shape of test as airport_wtc_multimodel, different axis: chains RUNWAY
into the winning blend as a finer WEIGHT (one extra least-squares scalar per
cell), not a nested trained booster. A nested per-(airport, exact runway code)
BOOSTER was checked and rejected before building anything -- RUNWAY_mvt's
per-airport cardinality is highly skewed (some airports show entire wake
categories at zero count on specific runways) and would very likely reproduce
the per_airport_flighttype failure (thin-cell overfit, worst at LIRF), only
with more, thinner cells. "Primary runway" (that airport's single most-used
runway code in train) vs "Other" collapses that cardinality to 2 buckets,
checked empirically before building: the primary-runway share ranges from
30% (EHAM) to 90% (LIRF) across the 10 airports, so every (airport, bucket)
cell is comfortably populated -- no bucket is a rounding error the way
per_airport_flighttype's 'Other' cells sometimes were.

Note this is a genuinely different real-world signal from
airport_wtc_multimodel: WTC asks "is this aircraft heavy/super", runway asks
"did this flight use the airport's main runway or a secondary one" -- runway
assignment reflects ATC/wind/traffic-flow decisions, only loosely tied to
aircraft size. Both are cheap to test (reuse of airport_operator_blend and
mlp_antigravity, no retraining) so there is no reason to guess which one
wins rather than measuring both.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_per_airport.model import AIRPORT_COL

ARM = "airport_runway_multimodel"
RUNWAY_COL = "RUNWAY_mvt"
DEFAULT_MIN_CELL_ROWS = 1000


def _primary_runway_map(ap, rwy) -> dict:
    """Each airport's single most-used runway code, learned from the array
    passed in (caller must pass the TRAIN pool, never val/test, to avoid
    leaking the held-out surface's own runway mix into the bucket definition)."""
    df = pd.DataFrame({"ap": ap, "rwy": rwy})
    return df.groupby("ap")["rwy"].agg(lambda s: s.value_counts().idxmax()).to_dict()


def _bucket_key(ap, X, primary_map) -> np.ndarray:
    rwy = X[RUNWAY_COL].astype("string").fillna("NaN").to_numpy()
    is_primary = np.array([primary_map.get(a) == r for a, r in zip(ap, rwy)])
    bucket = np.where(is_primary, "Primary", "Other")
    return np.array(["{}|{}".format(a, b) for a, b in zip(ap, bucket)])


def solve_alpha_per_cell(y, pa, pb, ap, X, primary_map, min_cell_rows) -> dict:
    airport_alpha = {}
    for a in sorted(pd.unique(ap)):
        m = (ap == a)
        d = pa[m] - pb[m]
        r = y[m] - pb[m]
        denom = float(np.dot(d, d))
        airport_alpha[str(a)] = 0.5 if denom <= 1e-12 else float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))

    cell = _bucket_key(ap, X, primary_map)
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


def blend_with_cell_alpha(pa, pb, ap, X, primary_map, cell_alpha, airport_alpha) -> np.ndarray:
    cell = _bucket_key(ap, X, primary_map)
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

    Xtr, _, meta_tr = D.load_part(args.split, "train", args.featureset, filter_target=True,
                                  outlier_policy=args.outlier_policy)
    primary_map = _primary_runway_map(meta_tr[AIRPORT_COL].astype("object").to_numpy(),
                                      Xtr[RUNWAY_COL].astype("string").fillna("NaN").to_numpy())
    print(f"[{ARM}] primary runway per airport (learned from TRAIN only): {primary_map}")

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
        yw, pa_w, pb_w, ap_w, Xw, primary_map, args.min_cell_rows)

    print(f"[{ARM}] solved cell weights (weight on airport_operator_blend):")
    for k, v in sorted(cell_alpha.items()):
        flag = " (fallback to airport alpha)" if k in fallback_cells else ""
        print(f"  {k:20s}: {v:.4f}{flag}")

    inner_blend = np.clip(blend_with_cell_alpha(pa_w, pb_w, ap_w, Xw, primary_map, cell_alpha, airport_alpha),
                          config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
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
        p_val_blend = np.clip(
            blend_with_cell_alpha(pa_val, pb_val, ap_val, Xva, primary_map, cell_alpha, airport_alpha),
            config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, p_val_blend)
        print(f"[{ARM}] Forward Validation Blend RMSE: {val_rmse:.4f}")

    meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "runway_col": RUNWAY_COL,
        "primary_runway_map": primary_map,
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
