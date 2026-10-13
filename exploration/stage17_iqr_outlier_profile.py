#!/usr/bin/env python3
"""Stage 17 -- IQR-based outlier profiling, as an EDA diagnostic (not a training
filter). Per user request: use IQR to flag anomalous taxi-out rows and profile
what's different about them, to surface NEW hypotheses -- not to touch the
training pool. `config.OUTLIER_POLICIES` already exhaustively tested fixed-bound
winsorize/drop filters and found `keep_all` (no filtering) wins every time
(stage6_clip_sweep.py); this script does not revisit that question.

Method, per airport (taxi-out distributions differ too much airport-to-airport
for a single global fence to mean anything, same reasoning as every other
stage in this doc)
  1. Q1, Q3, IQR = Q3-Q1 on that airport's DEP taxi-out (impossible values
     already dropped, y>0 only -- everything else kept, per this build's own
     "the tail is structured, not noise" finding).
  2. Tukey fences at 1.5xIQR (mild) and 3.0xIQR (extreme).
  3. Profile flagged vs not-flagged rows on covariates already known to matter
     in this doc: has_flight_record, hour, weekday, market segment (incl.
     Business), flight rule (IFR/VFR), operator (rate among the slow LIRF
     operator found in stage16), and the capacity utilization flag from stage14.
     This cross-checks new findings against everything already established,
     rather than rediscovering e.g. the no-flight-record tail effect from
     PLAN.md finding I under a new name.

Usage
    python3 stage17_iqr_outlier_profile.py
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[1]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common import io as cio
from common.bootstrap import write_json_guarded

OUT_DIR = config.ARTIFACTS / "audit"
AIRPORT_COL = "ADEP_mvt"

# The slow LIRF operator found in stage16_operator_effects.py (stand-residualized
# eta^2 = 0.103, this operator ran +1448s vs baseline) -- reused here to check
# whether it's over-represented among IQR outliers, not re-derived.
LIRF_SLOW_OPERATOR = "5689a9068938a8902f5282c8cc76c2a72e573c51773233abed420a52d6dc6e2c"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    X = D.build_scope("train2025", args.featureset)
    raw = cio.load_training(columns=[config.ID, "PHASE_mvt", config.TARGET,
                                     "MARKET_SEGMENT_flt", "FLIGHT_RULE_mvt",
                                     "AIRCRAFT_OPERATOR_flt"])
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    y_by_id = pd.Series(dep[config.TARGET].astype("float64").to_numpy(),
                        index=dep[config.ID].to_numpy())
    seg_by_id = pd.Series(dep["MARKET_SEGMENT_flt"].to_numpy(), index=dep[config.ID].to_numpy())
    rule_by_id = pd.Series(dep["FLIGHT_RULE_mvt"].to_numpy(), index=dep[config.ID].to_numpy())
    op_by_id = pd.Series(dep["AIRCRAFT_OPERATOR_flt"].to_numpy(), index=dep[config.ID].to_numpy())

    ids = X[config.ID].to_numpy()
    y = y_by_id.reindex(ids).to_numpy()
    segment = seg_by_id.reindex(ids).to_numpy()
    rule = rule_by_id.reindex(ids).to_numpy()
    operator = op_by_id.reindex(ids).to_numpy()

    valid = np.isfinite(y) & (y > config.TARGET_IMPOSSIBLE_MAX_SEC)
    airport = X[AIRPORT_COL].astype("object").to_numpy()
    hour = X["hour"].to_numpy()
    dow = X["dow"].to_numpy()
    has_record = X["has_flight_record"].to_numpy()
    high_util = X["high_util_flag"].to_numpy() if "high_util_flag" in X.columns else None

    rows = []
    all_flagged_mild = np.zeros(len(y), dtype=bool)
    for ap_code in sorted(pd.unique(airport[valid])):
        m = valid & (airport == ap_code)
        yy = y[m]
        q1, q3 = np.percentile(yy, [25, 75])
        iqr = q3 - q1
        lo_mild, hi_mild = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        lo_ext, hi_ext = q1 - 3.0 * iqr, q3 + 3.0 * iqr

        mild = (yy < lo_mild) | (yy > hi_mild)
        mild_lo = yy < lo_mild
        mild_hi = yy > hi_mild
        ext = (yy < lo_ext) | (yy > hi_ext)
        idx = np.flatnonzero(m)
        all_flagged_mild[idx[mild]] = True

        def profile(mask):
            n = int(mask.sum())
            if n == 0:
                return {"n": 0}
            sub_has_record = has_record[m][mask]
            sub_seg = segment[m][mask]
            sub_rule = rule[m][mask]
            sub_hour = hour[m][mask]
            sub_dow = dow[m][mask]
            sub_op = operator[m][mask]
            return {
                "n": n,
                "pct_of_airport": float(100.0 * n / len(yy)),
                "mean_taxiout": float(yy[mask].mean()),
                "pct_no_flight_record": float(100.0 * (sub_has_record == 0).mean()),
                "pct_business_segment": float(100.0 * pd.Series(sub_seg).eq("Business").mean()),
                "pct_vfr": float(100.0 * pd.Series(sub_rule).eq("V").mean()),
                "pct_slow_lirf_operator": (float(100.0 * pd.Series(sub_op).eq(LIRF_SLOW_OPERATOR).mean())
                                           if ap_code == "LIRF" else None),
                "pct_high_util_flag": (float(100.0 * (high_util[m][mask] == 1).mean())
                                       if high_util is not None else None),
                "top_hours": pd.Series(sub_hour).value_counts().head(3).to_dict(),
                "weekend_pct": float(100.0 * (sub_dow >= 5).mean()),
            }

        baseline_mask = ~mild  # non-outlier rows, for comparison
        rows.append({
            "airport": ap_code, "n_total": len(yy),
            "q1": float(q1), "q3": float(q3), "iqr": float(iqr),
            "fence_mild": [float(lo_mild), float(hi_mild)],
            "fence_extreme": [float(lo_ext), float(hi_ext)],
            "mild_outliers": profile(mild),
            "mild_outliers_lower_fast": profile(mild_lo),
            "mild_outliers_upper_slow": profile(mild_hi),
            "extreme_outliers": profile(ext),
            "baseline_non_outliers": profile(baseline_mask),
        })

    print("=" * 100)
    print("IQR outlier rate and profile, per airport (Tukey 1.5x mild / 3.0x extreme)")
    print("=" * 100)
    print("{:6s} {:>9s} {:>9s} {:>9s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "adep", "n", "%mild", "%ext", "norec_mild", "norec_base", "biz_mild", "biz_base"))
    for r in rows:
        mo, bo = r["mild_outliers"], r["baseline_non_outliers"]
        eo = r["extreme_outliers"]
        print("{:6s} {:9d} {:8.2f}% {:8.2f}% {:9.1f}% {:9.1f}% {:9.1f}% {:9.1f}%".format(
            r["airport"], r["n_total"], mo["pct_of_airport"], eo["pct_of_airport"],
            mo["pct_no_flight_record"], bo["pct_no_flight_record"],
            mo["pct_business_segment"], bo["pct_business_segment"]))

    print("\nLIRF slow-operator share (stage16 finding), upper(slow)/lower(fast)/baseline:")
    lirf = next(r for r in rows if r["airport"] == "LIRF")
    print("  upper(slow) outliers: {:.2f}%  |  lower(fast) outliers: {:.2f}%  |  baseline: {:.2f}%".format(
        lirf["mild_outliers_upper_slow"]["pct_slow_lirf_operator"],
        lirf["mild_outliers_lower_fast"]["pct_slow_lirf_operator"],
        lirf["baseline_non_outliers"]["pct_slow_lirf_operator"]))

    print("\nLEMD business-segment share, upper(slow)/lower(fast)/baseline (S1b found LEMD")
    print("business flights run SLOWER even stand-controlled -- checking if this is why):")
    lemd = next(r for r in rows if r["airport"] == "LEMD")
    print("  upper(slow): {:.1f}%  |  lower(fast): {:.1f}%  |  baseline: {:.1f}%".format(
        lemd["mild_outliers_upper_slow"]["pct_business_segment"],
        lemd["mild_outliers_lower_fast"]["pct_business_segment"],
        lemd["baseline_non_outliers"]["pct_business_segment"]))

    print("\nVFR share (1c finding), upper(slow)/lower(fast)/baseline, EHAM/LSZH:")
    for ap_code in ("EHAM", "LSZH"):
        r = next(r for r in rows if r["airport"] == ap_code)
        print("  {}: upper(slow) {:.2f}% VFR | lower(fast) {:.2f}% VFR | baseline {:.2f}% VFR".format(
            ap_code, r["mild_outliers_upper_slow"]["pct_vfr"],
            r["mild_outliers_lower_fast"]["pct_vfr"], r["baseline_non_outliers"]["pct_vfr"]))

    print("\nhigh_util_flag share (S0 finding), upper(slow)/lower(fast)/baseline, EGLL/LSZH:")
    for ap_code in ("EGLL", "LSZH"):
        r = next(r for r in rows if r["airport"] == ap_code)
        print("  {}: upper(slow) {:.1f}% | lower(fast) {:.1f}% | baseline {:.1f}%".format(
            ap_code, r["mild_outliers_upper_slow"]["pct_high_util_flag"],
            r["mild_outliers_lower_fast"]["pct_high_util_flag"], r["baseline_non_outliers"]["pct_high_util_flag"]))

    print("\nOutlier direction split (n), all airports:")
    for r in rows:
        print("  {:6s} lower(fast)={:6d}  upper(slow)={:6d}".format(
            r["airport"], r["mild_outliers_lower_fast"]["n"], r["mild_outliers_upper_slow"]["n"]))

    report = {"featureset": args.featureset, "airports": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_guarded(OUT_DIR / "iqr_outlier_profile.json", report, force=args.force)
    print("\nwrote {}".format(OUT_DIR / "iqr_outlier_profile.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
