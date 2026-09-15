#!/usr/bin/env python3
"""Stage 14 -- test TAXI_OUT_DRIVERS.md hypothesis 0: capacity regime (L1/L2/L3).

v3 (advisor review, second pass): v1/v2 both used Spearman rho(queue, taxi-out)
as the regime signal. That is WRONG in principle, not just miscalibrated: rho
measures a RESPONSE SLOPE (how much taxi-out moves per unit of queue), while
"L1/L2/L3" is a LEVEL (demand vs. capacity). A genuinely saturated runway sits
on a plateau -- everyone waits a long time regardless of exactly how deep the
queue is -- so rho COLLAPSES toward zero exactly where saturation is worst.
v2's finding that peak-only rho was weaker than pooled at most airports (e.g.
LSZH peak -0.214, EHAM peak 0.028) was mis-read as a restriction-of-range
artifact; it is more likely the saturation plateau itself -- the strongest
capacity evidence in the data, upside down under a slope metric.

v3 measures a LEVEL instead, with no external capacity figures needed:

  1. Demand proxy: `dep_prev_60m + arr_prev_60m` (total runway-relevant
     movement count in the trailing hour) -- both columns already exist from
     common/features.py's congestion builder.
  2. Observed capacity ceiling, per airport: the 95th percentile of that
     demand proxy. Empirical, self-normalized -- this dataset ships no
     declared runway capacity figure to calibrate against.
  3. Utilization = demand / ceiling. `frac_high_util` = share of an airport's
     rows running at >=90% of its OWN ceiling -- how often it operates near
     its historical max, a level, not a slope.
  4. Plateau check: bin utilization into quintiles per airport, look at the
     mean-taxi-out step from bin 4->5 vs the average of the earlier steps. A
     small/negative top step is the saturation signature this hypothesis
     actually predicts.
  5. Regime combines both: rarely-high-utilization -> L1; often near ceiling
     AND a flattening top step -> L3; everything else -> L2.

v1/v2's rho tables are kept in the JSON (renamed `response_slope_rho`,
`regime_label` REMOVED from them) as real queue-response measurements -- they
answer "how legible is the queuing mechanism here," a different and still
possibly useful question, just not this one. Nothing in TAXI_OUT_DRIVERS.md
should cite an L1/L2/L3 label derived from rho.

Usage
    python3 stage14_capacity_regime.py
    python3 stage14_capacity_regime.py --ceiling-pctl 99 --force
"""
from __future__ import annotations

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
QUEUE_COLS_FOR_SLOPE = ["same_rwy_prev_15m", "dep_prev_15m"]
DEMAND_COLS = ["dep_prev_60m", "arr_prev_60m"]
CEILING_PCTL_DEFAULT = 95.0
HIGH_UTIL_THRESHOLD = 0.90
N_BINS = 5

L1_MAX_FRAC_HIGH_UTIL = 0.05   # rarely runs near its own ceiling
L3_MIN_FRAC_HIGH_UTIL = 0.15   # often runs near its own ceiling
L3_MAX_PLATEAU_RATIO = 0.30    # top bin's step must be <=30% of the earlier average step


def _rho(a: "np.ndarray", b: "np.ndarray") -> float:
    if len(a) < 20:
        return float("nan")
    return float(pd.Series(np.asarray(a)).corr(pd.Series(np.asarray(b)), method="spearman"))


def utilization_regime(yy, util, n_bins=N_BINS):
    """Level-based regime: how often near the ceiling, and does it plateau there?"""
    frac_high = float((util >= HIGH_UTIL_THRESHOLD).mean())

    try:
        bins = pd.qcut(util, n_bins, duplicates="drop")
    except ValueError:
        return frac_high, None, "P", []

    df = pd.DataFrame({"bin": bins, "y": yy})
    means = df.groupby("bin", observed=True)["y"].mean()
    means = means.reindex(sorted(means.index, key=lambda iv: iv.left))
    bin_means = [float(v) for v in means.to_numpy()]

    if len(bin_means) < 3:
        plateau_ratio = None
    else:
        steps = np.diff(bin_means)
        top_step = steps[-1]
        earlier_avg = float(np.mean(steps[:-1])) if len(steps) > 1 else steps[0]
        plateau_ratio = float(top_step / earlier_avg) if earlier_avg > 0 else None

    if frac_high < L1_MAX_FRAC_HIGH_UTIL:
        label = "L1"
    elif frac_high >= L3_MIN_FRAC_HIGH_UTIL and plateau_ratio is not None and plateau_ratio <= L3_MAX_PLATEAU_RATIO:
        label = "L3"
    else:
        label = "L2"
    return frac_high, plateau_ratio, label, bin_means


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--ceiling-pctl", type=float, default=CEILING_PCTL_DEFAULT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    X = D.build_scope("train2025", args.featureset)
    raw = cio.load_training(columns=[config.ID, "PHASE_mvt", config.TARGET])
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    y_by_id = pd.Series(dep[config.TARGET].astype("float64").to_numpy(),
                        index=dep[config.ID].to_numpy())

    ids = X[config.ID].to_numpy()
    y = y_by_id.reindex(ids).to_numpy()
    airport = X[AIRPORT_COL].astype("object").to_numpy()
    demand = sum(X[c].to_numpy(dtype="float64") for c in DEMAND_COLS)
    slope_cols = {c: X[c].to_numpy(dtype="float64") for c in QUEUE_COLS_FOR_SLOPE}

    valid = np.isfinite(y) & (y > config.TARGET_IMPOSSIBLE_MAX_SEC)
    print("dropped {} rows with impossible/missing target out of {}".format(
        int((~valid).sum()), len(valid)))

    airport_rows = []
    for ap_code in sorted(pd.unique(airport[valid])):
        m = valid & (airport == ap_code)
        yy = y[m]
        dd = demand[m]
        ceiling = float(np.percentile(dd, args.ceiling_pctl))
        util = dd / ceiling if ceiling > 0 else dd * 0.0

        frac_high, plateau_ratio, label, bin_means = utilization_regime(yy, util)

        row = {
            "airport": ap_code, "n": int(m.sum()), "mean_taxiout": float(yy.mean()),
            "ceiling_pctl": args.ceiling_pctl, "ceiling_value": ceiling,
            "frac_time_high_utilization": frac_high,
            "plateau_ratio_top_bin": plateau_ratio,
            "utilization_bin_means_taxiout": bin_means,
            "regime_label": label,
        }
        for qc in QUEUE_COLS_FOR_SLOPE:
            row["response_slope_rho__{}".format(qc)] = _rho(slope_cols[qc][m], yy)
        airport_rows.append(row)

    airport_rows.sort(key=lambda r: -r["frac_time_high_utilization"])

    print("=" * 112)
    print("Capacity regime by airport -- LEVEL-based (utilization vs. own {:.0f}th-pctl ceiling)".format(
        args.ceiling_pctl))
    print("=" * 112)
    print("{:6s} {:>8s} {:>9s} {:>10s} {:>10s} {:>10s} {:>4s}".format(
        "adep", "n", "mean_y", "frac_hi_u", "plateau", "ceiling(n)", "L?"))
    for r in airport_rows:
        print("{:6s} {:8d} {:9.1f} {:9.1%} {:>10s} {:10.1f} {:>4s}".format(
            r["airport"], r["n"], r["mean_taxiout"], r["frac_time_high_utilization"],
            "{:.2f}".format(r["plateau_ratio_top_bin"]) if r["plateau_ratio_top_bin"] is not None else "n/a",
            r["ceiling_value"], r["regime_label"]))

    print("\n  frac_hi_u = share of rows running at >={:.0%} of the airport's OWN 95th-pctl".format(
        HIGH_UTIL_THRESHOLD))
    print("  demand ceiling (dep_prev_60m + arr_prev_60m). plateau = (last bin's taxi-out")
    print("  step) / (average of the earlier steps) -- near 0 or negative means taxi-out")
    print("  stops climbing at the top of the utilization range (a saturation signature).")
    print("  L1: frac_hi_u < {:.0%} (rarely near its own ceiling)".format(L1_MAX_FRAC_HIGH_UTIL))
    print("  L3: frac_hi_u >= {:.0%} AND plateau <= {:.2f} (often near ceiling, flattens there)".format(
        L3_MIN_FRAC_HIGH_UTIL, L3_MAX_PLATEAU_RATIO))
    print("  L2: everything else")

    print("\n  (kept for reference, NOT used for the regime label -- rho is a response")
    print("   SLOPE, not a saturation LEVEL; a real plateau makes rho fall, not rise)")
    print("  {:6s} {:>9s} {:>9s}".format("adep", "rho_rwy15", "rho_dep15"))
    for r in sorted(airport_rows, key=lambda r: r["airport"]):
        print("  {:6s} {:9.3f} {:9.3f}".format(
            r["airport"], r["response_slope_rho__same_rwy_prev_15m"],
            r["response_slope_rho__dep_prev_15m"]))

    report = {
        "featureset": args.featureset, "demand_cols": DEMAND_COLS,
        "ceiling_pctl": args.ceiling_pctl, "high_util_threshold": HIGH_UTIL_THRESHOLD,
        "l1_max_frac_high_util": L1_MAX_FRAC_HIGH_UTIL,
        "l3_min_frac_high_util": L3_MIN_FRAC_HIGH_UTIL,
        "l3_max_plateau_ratio": L3_MAX_PLATEAU_RATIO,
        "airports": airport_rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_guarded(OUT_DIR / "capacity_regime.json", report, force=args.force)
    print("\nwrote {}".format(OUT_DIR / "capacity_regime.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
