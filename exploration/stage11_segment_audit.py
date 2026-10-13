#!/usr/bin/env python3
"""Stage 11: loss-surface + segment-viability audit for one arm.

Two things every future "let's segment on X" idea should go through before a
pipeline gets built around it -- this stage automates both:

1. The per-airport loss surface (reuses common/evaluate's own numbers, so this
   never disagrees with an arm's own eval_*.json -- it just also prints them
   without needing to re-run training).
2. A segment-viability check for any named boolean condition: how many rows
   actually exist in the full year / the honest fitting surface / real val /
   real test, and (if a "this is basically deterministic" claim is being
   made) whether the residual variance backs that up.

Built directly in response to auditing pipeline_airport_debiased_multimodel's
LIRF dual-regime alignment, whose two segments turned out to be very different
in kind: the 25,200-50,000s band is a near-exact telemetry identity (34
rows/year, residual std 2.9s) and safe to hard-code with zero free parameters;
the >50,000s band is NOT (20 rows/year, residual std 13,488s) and any fitted
correction there is fitting noise, not a discovered pattern, however good the
resulting number looks on the 3-4 test rows it touches. See
plans/11_plan_outlier_audit_and_overfitting_check.md for the full writeup.

Usage:
    python3 stage11_segment_audit.py --arm airport_multiseed_multimodel
    python3 stage11_segment_audit.py --arm airport_debiased_multimodel --skip-lirf-bands
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common.predict import predict_arm
from common.segment_audit import segment_report


def print_loss_surface(arm: str, split: str, featureset: str, seed: int, outlier_policy: str):
    for part in ("val", "test"):
        proto = config.protocol_key(split, featureset, outlier_policy)
        run = config.ARTIFACTS / "runs" / "{}__{}__seed{}".format(arm, proto, seed)
        p = run / "eval_{}.json".format(part)
        if not p.exists():
            print("  (no eval_{}.json for {} -- skipping)".format(part, arm))
            continue
        res = json.loads(p.read_text())
        by_ap = pd.DataFrame(res["rmse_by_airport"]).T.sort_values("rmse")
        total_sse = float((by_ap["n"] * by_ap["rmse"] ** 2).sum())
        print("\n== {} / {} -- loss surface ==".format(arm, part))
        print("  pooled RMSE: {:.2f} s   LB-weighted: {}".format(
            res["rmse"],
            "{:.2f} s".format(res["leaderboard_weighted_rmse"])
            if res.get("leaderboard_weighted_rmse") else "n/a"))
        print("  {:6s} {:>8s} {:>10s} {:>10s} {:>10s}".format(
            "AP", "n", "RMSE", "mean_y", "SSE share"))
        for k, r in by_ap.sort_values("rmse", ascending=False).iterrows():
            sse = r["n"] * r["rmse"] ** 2
            print("  {:6s} {:8.0f} {:10.2f} {:10.1f} {:9.1%}".format(
                str(k), r["n"], r["rmse"], r["mean_true"], sse / total_sse if total_sse else 0.0))
        print("  worst 0.1% / 1% of rows carry {:.1%} / {:.1%} of total SSE".format(
            res["sse_share_worst_0.1pct"], res["sse_share_worst_1pct"]))


def audit_lirf_delay_bands(featureset: str, outlier_policy: str):
    """The specific check that would have flagged airport_debiased_multimodel's
    extreme-band fit as high-risk before it was built, not after."""
    print("\n== LIRF no-flight-record delay-band viability (delta_sched_sec) ==")
    Xf, yf, mf = D.load_part("full", "train", featureset, filter_target=True,
                             outlier_policy=outlier_policy)
    apf = mf["ADEP_mvt"].astype("object").to_numpy()
    hrf = Xf["has_flight_record"].to_numpy()
    dsf = Xf["delta_sched_sec"].fillna(0).to_numpy()
    yf = np.asarray(yf, dtype="float64")
    base_f = (apf == "LIRF") & (hrf == 0)

    surfaces = {}
    for split, part in [("forward_inner", "val"), ("forward", "val"), ("forward", "test")]:
        X, y, m = D.load_part(split, part, featureset, filter_target=False)
        ap = m["ADEP_mvt"].astype("object").to_numpy()
        hr = X["has_flight_record"].to_numpy()
        ds = X["delta_sched_sec"].fillna(0).to_numpy()
        surfaces[(split, part)] = ((ap == "LIRF") & (hr == 0), ds)

    bands = [("mid 25.2k-50k s (7-14h)", 25200, 50000, 3), ("extreme >50k s (>14h)", 50000, 1e9, 3)]
    for label, lo, hi, n_params in bands:
        seg_f = base_f & (dsf > lo) & (dsf <= hi)
        m_fi, ds_fi = surfaces[("forward_inner", "val")]
        m_v, ds_v = surfaces[("forward", "val")]
        m_t, ds_t = surfaces[("forward", "test")]
        seg_fi = m_fi & (ds_fi > lo) & (ds_fi <= hi)
        seg_v = m_v & (ds_v > lo) & (ds_v <= hi)
        seg_t = m_t & (ds_t > lo) & (ds_t <= hi)
        segment_report(
            label, n_full_year=int(seg_f.sum()), n_fit_surface=int(seg_fi.sum()),
            n_val=int(seg_v.sum()), n_test=int(seg_t.sum()), n_free_params=n_params,
            residual=(yf[seg_f] - dsf[seg_f]), residual_label="y - delta_sched_sec")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="airport_multiseed_multimodel")
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--skip-lirf-bands", action="store_true",
                    help="skip the LIRF delay-band viability check")
    args = ap.parse_args(argv)

    print_loss_surface(args.arm, args.split, args.featureset, args.seed, args.outlier_policy)
    if not args.skip_lirf_bands:
        audit_lirf_delay_bands(args.featureset, args.outlier_policy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
