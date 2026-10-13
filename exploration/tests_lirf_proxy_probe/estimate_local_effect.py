#!/usr/bin/env python3
"""Diagnostic (no training): what would a POOLED board score look like if
LIRF's AOBT-unmatched rows were overridden with the plain schedule proxy
(delta_sched_sec = MVT_TIME - SCHED_TIME) instead of the shipped model's
prediction?

Motivation: SS2q/SS2m already tested a FITTED convex mix of model and proxy
on this population and found it unstable across surfaces (won 2/3, lost
decisively on the board-matched one) -- but that was a grid-searched blend
on 534 rows, a different (and overfit-prone) object from a plain, unfitted
swap. `idband` (a related but different intervention: routing flagged rows
to a fallback CONSTANT) was submitted live twice and regressed both times,
but only by +0.86s/+0.87s -- small. This checks the plain swap's pooled
effect directly on both forward and seasonal_full test, cleanly, before
deciding whether it's worth a live submission slot as a probe.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common.predict import predict_arm

FEATURESET, OUTLIER_POLICY = "with_aobt", "keep_all"
ARM, SEED, TAG = "airport_operator_multimodel", 42, "msmlp"


def rmse(err):
    return float(np.sqrt(np.mean(err ** 2)))


def check(split: str, part: str) -> None:
    print("\n" + "=" * 70)
    print("[lirf-proxy-probe] split={} part={}".format(split, part))
    X, y, meta = D.load_part(split, part, FEATURESET, filter_target=False)
    y = np.asarray(y, dtype="float64")
    p = predict_arm(ARM, split, part, FEATURESET, SEED, OUTLIER_POLICY,
                     X=X, meta=meta, tag=TAG)

    airport = (meta["ADEP_mvt"].astype(str).to_numpy()
               if "ADEP_mvt" in meta.columns else X["ADEP_mvt"].astype(str).to_numpy())
    is_lirf = airport == "LIRF"
    is_unmatched = ~X["has_aobt"].to_numpy().astype(bool)
    target_mask = is_lirf & is_unmatched
    print("  n_total={}  LIRF={}  LIRF-unmatched (probe target)={} ({:.3%} of all rows)".format(
        len(y), is_lirf.sum(), target_mask.sum(), target_mask.mean()))

    proxy = X["delta_sched_sec"].to_numpy()
    p_probe = p.copy()
    p_probe[target_mask] = np.clip(
        proxy[target_mask], config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)

    err_base = y - p
    err_probe = y - p_probe

    rmse_base_pooled = rmse(err_base)
    rmse_probe_pooled = rmse(err_probe)
    rmse_base_target = rmse(err_base[target_mask])
    rmse_probe_target = rmse(err_probe[target_mask])

    print("  target population: model RMSE = {:.2f}   proxy RMSE = {:.2f}   "
          "(negative = proxy worse)".format(rmse_base_target, rmse_probe_target))
    print("  POOLED: base (model) = {:.4f}   probe (proxy-override) = {:.4f}   "
          "delta = {:+.4f}s  (negative = probe better)".format(
              rmse_base_pooled, rmse_probe_pooled, rmse_probe_pooled - rmse_base_pooled))


def main() -> int:
    check("forward", "val")
    check("forward", "test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
