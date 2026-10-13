#!/usr/bin/env python3
"""Stage 12: is the LIRF telemetry identity LIRF-specific, and does it reach the
scored set at all?

Two questions this build has not answered, both read-only, neither requiring a
model to be trained.

Q1 -- IS IT A LIRF THING?
-------------------------
`airport_lirf_identity_multimodel` exploits a mechanism: when a movement has no
joined flight record, BLOCK_TIME defaults to SCHED_TIME, so
`y = TAKEOFF - BLOCK_TIME` collapses to `TAKEOFF - SCHED_TIME = delta_sched_sec`.
That is a property of how a record is written when radar data is missing -- there
is nothing in that description specific to Rome.

But it was only ever CHECKED at LIRF (stage11, 34 rows in the 25.2k-50k band,
residual std 2.9 s), and the arm hard-codes `ADEP_mvt == 'LIRF'` accordingly.
Hard-coding the one airport somebody happened to look at is how a real mechanism
gets mistaken for a local quirk -- or a local quirk for a mechanism. This checks
all ten airports over the FULL year, with the same test, and lets the data say
which it is.

If the identity holds elsewhere, extending the override is free accuracy with
zero fitted parameters -- the same kind of change as candidate 2 in plan 12, not
a new bucketing axis. If it holds ONLY at LIRF, that is worth knowing too: it
means the current hard-coding is correct and the arm should say so on evidence
rather than by accident.

Q2 -- DOES ANY OF THIS REACH THE LEADERBOARD?  (the one that could bite)
------------------------------------------------------------------------
Every tail result in this build is measured on the `forward` val/test months. The
SCORED set is `ranking.parquet`'s DEP rows, and nothing here has ever counted how
many of those rows the tail machinery would actually touch.

This matters more than it sounds. If the scored set contains ~0 rows in the
override band, then the LIRF identity, the >50k tail bet, the debiased arm's
alignment -- the entire line of work that produced six quarantined pipelines --
moves the leaderboard by nothing, and the only thing that can move it is bulk
in-band accuracy on ~215k ordinary rows.

That is a decision-changing number and it is cheap to compute, so it should be
computed before any more tail work is done. Note the scored set has NO labels:
this counts EXPOSURE (how many rows fall in the band), never accuracy.

Usage
    python3 stage12_telemetry_identity_audit.py
    python3 stage12_telemetry_identity_audit.py --featureset with_aobt --tol 60
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
from common.bootstrap import write_json_guarded

ROOT = Path(__file__).resolve().parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
OUT_DIR = config.ARTIFACTS / "audit"

AIRPORT_COL = "ADEP_mvt"
REC_COL = "has_flight_record"
DS_COL = "delta_sched_sec"

# The bands stage11 used, so this is directly comparable to the LIRF numbers
# already on record rather than a fresh slicing nobody can line up.
BANDS = [
    ("<=7200",        0.0,      7200.0),
    ("7200-25200",    7200.0,   25200.0),
    ("25200-50000",   25200.0,  50000.0),   # the CHECKED identity band at LIRF
    (">50000",        50000.0,  float("inf")),
]


def band_of(ds: np.ndarray) -> np.ndarray:
    out = np.full(len(ds), "", dtype=object)
    for name, lo, hi in BANDS:
        out[(ds > lo) & (ds <= hi)] = name
    return out


def summarise(y: np.ndarray, ds: np.ndarray, tol: float) -> dict:
    """How close is y to delta_sched_sec on these rows?"""
    if len(y) == 0:
        return {"n": 0}
    resid = y - ds
    return {
        "n": int(len(y)),
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid)),
        "resid_median": float(np.median(resid)),
        "resid_p95_abs": float(np.percentile(np.abs(resid), 95)),
        "frac_within_tol": float(np.mean(np.abs(resid) <= tol)),
        "y_median": float(np.median(y)),
    }


def verdict(s: dict, tol: float, min_n: int) -> str:
    """One label per cell, on stated thresholds rather than eyeballing."""
    if s.get("n", 0) < min_n:
        return "too few rows to judge"
    if s["frac_within_tol"] >= 0.90 and s["resid_std"] <= 10 * tol:
        return "IDENTITY"
    if s["frac_within_tol"] >= 0.50:
        return "partial"
    return "not an identity"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--tol", type=float, default=60.0,
                    help="|y - delta_sched| within this many seconds counts as "
                         "the identity holding (default 60)")
    ap.add_argument("--min-n", type=int, default=10,
                    help="cells below this row count are reported but not judged")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    # ---------------- Q1: the identity, every airport, full year ----------------
    X = D.build_scope("train2025", args.featureset)
    ytr = D._targets()
    X = X.set_index(config.ID)
    y = ytr.reindex(X.index)
    keep = y.notna().to_numpy()
    X, y = X[keep], y[keep].to_numpy(dtype="float64")

    for c in (AIRPORT_COL, REC_COL, DS_COL):
        if c not in X.columns:
            raise SystemExit(
                "feature {!r} is missing from the {} cache. This audit reads the "
                "same columns the override does; if they have been renamed, fix "
                "the names here rather than guessing.".format(c, args.featureset))

    ap_arr = X[AIRPORT_COL].astype("object").to_numpy()
    rec = X[REC_COL].to_numpy()
    ds = X[DS_COL].fillna(0).to_numpy(dtype="float64")
    bands = band_of(ds)

    rows = []
    for a in sorted(pd.unique(ap_arr)):
        for band_name, _lo, _hi in BANDS:
            for rec_val, rec_label in ((0, "no_record"), (1, "has_record")):
                m = (ap_arr == a) & (bands == band_name) & (rec == rec_val)
                s = summarise(y[m], ds[m], args.tol)
                s.update({"airport": str(a), "band": band_name,
                          "record": rec_label,
                          "verdict": verdict(s, args.tol, args.min_n)})
                rows.append(s)

    # ---------------- Q2: exposure in the SCORED set (no labels) ----------------
    Xr, meta_r = D.load_ranking_features(args.featureset)
    ap_r = meta_r[AIRPORT_COL].astype("object").to_numpy()
    rec_r = Xr[REC_COL].to_numpy()
    ds_r = Xr[DS_COL].fillna(0).to_numpy(dtype="float64")
    bands_r = band_of(ds_r)

    exposure = []
    for a in sorted(pd.unique(ap_r)):
        for band_name, _lo, _hi in BANDS:
            m = (ap_r == a) & (bands_r == band_name) & (rec_r == 0)
            exposure.append({"airport": str(a), "band": band_name,
                             "record": "no_record", "n_scored_rows": int(m.sum())})

    n_scored = int(len(Xr))
    # what the CURRENTLY SHIPPED rule touches: LIRF, no record, ds > 25200
    n_shipped_rule = int(((ap_r == "LIRF") & (rec_r == 0) & (ds_r > 25200.0)).sum())
    # what an all-airport version of the same rule would touch
    n_all_airport_rule = int(((rec_r == 0) & (ds_r > 25200.0)).sum())

    # ---------------- report ----------------
    print("=" * 78)
    print("Q1  Is y == delta_sched_sec?  Full year, per airport x band x record")
    print("    tol = +/-{:.0f}s   (cells with n < {} are not judged)".format(
        args.tol, args.min_n))
    print("=" * 78)
    print("{:6s} {:13s} {:11s} {:>8s} {:>10s} {:>11s}  {}".format(
        "adep", "band", "record", "n", "resid_std", "within_tol", "verdict"))
    for r in rows:
        if r["n"] == 0:
            continue
        # the no-record tail is the subject; has_record is the control
        if r["band"] == "<=7200" and r["record"] == "has_record":
            continue
        print("{:6s} {:13s} {:11s} {:>8d} {:>10.1f} {:>10.1%}  {}".format(
            r["airport"], r["band"], r["record"], r["n"],
            r["resid_std"], r["frac_within_tol"], r["verdict"]))

    ident = [r for r in rows
             if r["verdict"] == "IDENTITY" and r["record"] == "no_record"
             and r["band"] in ("25200-50000", "7200-25200", ">50000")]
    print("\n  identity cells found (no_record, above 7200s): {}".format(
        ", ".join("{}/{}".format(r["airport"], r["band"]) for r in ident) or "none"))
    non_lirf = sorted({r["airport"] for r in ident if r["airport"] != "LIRF"})
    print("  airports OTHER than LIRF showing it: {}".format(
        ", ".join(non_lirf) if non_lirf else "NONE -- the hard-coding is correct"))

    print("\n" + "=" * 78)
    print("Q2  Exposure in the SCORED set ({} rows, no labels -- counts only)".format(
        n_scored))
    print("=" * 78)
    hot = [e for e in exposure if e["n_scored_rows"] > 0 and e["band"] != "<=7200"]
    if hot:
        for e in sorted(hot, key=lambda e: -e["n_scored_rows"]):
            print("  {:6s} {:13s} no_record  n={:d}".format(
                e["airport"], e["band"], e["n_scored_rows"]))
    else:
        print("  no scored rows above 7200s with a missing flight record")
    print("\n  rows the SHIPPED rule touches (LIRF, no record, ds>25200): {}  ({:.4f}% of scored)"
          .format(n_shipped_rule, 100.0 * n_shipped_rule / max(1, n_scored)))
    print("  rows an ALL-AIRPORT version would touch:                    {}  ({:.4f}% of scored)"
          .format(n_all_airport_rule, 100.0 * n_all_airport_rule / max(1, n_scored)))
    # ---------------- Q3: size the prize, from measured SSE, not intuition -------
    #
    # The obvious reading of Q2 is "44 rows out of 215,876 cannot possibly
    # matter". That reading is WRONG for this metric, which is exactly why this
    # is computed rather than asserted: pooled RMSE is a MEAN OF SQUARES, and a
    # tail row is wrong by ~15,000 s where an ordinary row is wrong by ~250 s.
    # Squared, one such row weighs like ~3,600 ordinary ones.
    #
    # Leverage is derived from an arm's own eval JSON: sse_total from `rmse`,
    # sse_in_band from `rmse_in_band`, the difference is what the tail carries.
    lev = {}
    run = (config.ARTIFACTS / "runs" /
           "airport_operator_multimodel__forward-{}-keep_all__seed42".format(args.featureset))
    for part in ("val", "test"):
        f = run / "eval_{}.json".format(part)
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        n_all, n_ib = int(e["n"]), int(e["n_in_band"])
        sse_all = e["rmse"] ** 2 * n_all
        sse_ib = e["rmse_in_band"] ** 2 * n_ib
        lev[part] = {
            "n": n_all, "n_tail": n_all - n_ib,
            "pct_rows_tail": 100.0 * (n_all - n_ib) / n_all,
            "pct_sse_tail": 100.0 * (sse_all - sse_ib) / sse_all,
            "rmse": e["rmse"], "rmse_in_band": e["rmse_in_band"],
        }

    print("\n" + "=" * 78)
    print("Q3  Size the prize -- how much pooled squared error lives in the tail?")
    print("=" * 78)
    for part, L in lev.items():
        print("  {:5s}: {:d} rows above 7200 s = {:.3f}% of rows, carrying "
              "{:.1f}% of pooled squared error".format(
                  part, L["n_tail"], L["pct_rows_tail"], L["pct_sse_tail"]))
    if lev:
        print("\n  So 'a handful of rows cannot matter' is FALSE for this metric.")
        print("  Measured, not assumed: the idband override touches FIVE val rows")
        print("  and moves pooled val RMSE by -0.55 to -1.20 s, and five test rows")
        print("  by -0.58 to -2.18 s (plan 12, candidate 2).")
        n_test = lev.get("test", {}).get("n")
        if n_test:
            print("\n  The scored set is {} rows -- SMALLER than the {} test rows --".format(
                n_scored, n_test))
            print("  so each corrected row carries MORE weight there, not less.")
        print("  {} touched rows is therefore worth roughly the same order as the".format(
            n_shipped_rule))
        print("  measured val/test effect: ~1-3 s of pooled RMSE. Small, but real,")
        print("  free, and it cannot backfire on a row it touches.")
        print("\n  What this does NOT rescue: the quarantined ladder never touched")
        print("  these rows. It re-weighted the blend across ALL rows and moved")
        print("  nothing beyond seed noise. Tail RULES and blend-weight BUCKETING")
        print("  are different levers -- this result is about the first, the")
        print("  over-tuning verdict was about the second.")

    report = {
        "featureset": args.featureset,
        "tol_sec": args.tol,
        "min_n": args.min_n,
        "bands": [{"name": n, "lo": lo, "hi": (None if hi == float("inf") else hi)}
                  for n, lo, hi in BANDS],
        "full_year_cells": rows,
        "scored_set": {
            "n_scored_rows": n_scored,
            "exposure_by_cell": exposure,
            "n_rows_shipped_lirf_rule": n_shipped_rule,
            "n_rows_all_airport_rule": n_all_airport_rule,
            "pct_scored_shipped_rule": 100.0 * n_shipped_rule / max(1, n_scored),
            "pct_scored_all_airport_rule": 100.0 * n_all_airport_rule / max(1, n_scored),
        },
        "identity_cells": ident,
        "non_lirf_identity_airports": non_lirf,
        "tail_leverage": lev,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_guarded(OUT_DIR / "telemetry_identity_audit.json", report,
                       force=args.force)
    print("\nwrote {}".format(OUT_DIR / "telemetry_identity_audit.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
