#!/usr/bin/env python3
"""Stage 15 -- test TAXI_OUT_DRIVERS.md hypothesis 1: priority rules.

Two sub-hypotheses:

  (a) Arrivals are prioritized over departures, so departure taxi-out should
      track INBOUND traffic (arr_prev_*m) at least as strongly as it tracks
      outbound queuing (dep_prev_*m / same_rwy_prev_*m from stage14).

  (b) Business/priority aircraft get parking priority but NOT departure
      priority, so MARKET_SEGMENT_flt == "Business" should show little or no
      taxi-out advantage over other segments once stand/runway geometry is
      held roughly constant (features.py: "stand->runway is the single
      strongest geometric determinant of taxi distance" -- a raw mean
      comparison ignoring that conflates "priority treatment" with "business
      jets sit on different stands").

Both checks reuse existing congestion/categorical columns from
common/features.py -- no new features are built here, this is descriptive EDA.

Usage
    python3 stage15_priority_rules.py
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
RUNWAY_COL = "RUNWAY_mvt"
STAND_COL = "STAND_mvt"
SEGMENT_COL = "MARKET_SEGMENT_flt"
IN_BAND_MAX = 7200.0


def part_a2() -> list:
    """Direct operational reading of 1a: is taxi-TO-departure longer than
    taxi-TO-arrival, per airport?

    Correction (user, after part_a): part_a tested a volume-correlation proxy
    ("does departure delay track arrival traffic"), which cannot see an ATC
    sequencing/priority procedure -- that's operational, not a traffic-count
    effect. What was actually meant is the direct duration comparison: do
    departures spend longer getting to the runway than arrivals spend getting
    to the gate. TAXITIME_SEC_mvt turns out to be populated on BOTH phases
    (verified: ARR rows are non-null, mean 536.8s vs DEP mean 991.4s pooled) --
    common/features.py already relies on this for `arr_taxiin_mean_prev_60m` --
    so this is a direct read of the same target column, not a new proxy.

    Caveat: a longer taxi-OUT than taxi-IN is consistent with "departures wait
    on arrivals," but is not proof of it -- departure routes plausibly involve
    more distance/holding (de-icing pads, remote stands, runway-entry queuing
    for wake separation) for reasons that have nothing to do with arrival
    priority. This measures the operational asymmetry the hypothesis predicts;
    it does not isolate the CAUSE of that asymmetry.
    """
    raw = cio.load_training(columns=[config.ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", config.TARGET])
    y = raw[config.TARGET].astype("float64").to_numpy()
    valid = np.isfinite(y) & (y > config.TARGET_IMPOSSIBLE_MAX_SEC)
    is_dep = raw["PHASE_mvt"].eq("DEP").to_numpy()
    airport = np.where(is_dep, raw["ADEP_mvt"].to_numpy(), raw["ADES_mvt"].to_numpy())

    rows = []
    for ap_code in sorted(pd.unique(airport[valid])):
        m_dep = valid & is_dep & (airport == ap_code)
        m_arr = valid & ~is_dep & (airport == ap_code)
        if m_dep.sum() == 0 or m_arr.sum() == 0:
            continue
        y_dep, y_arr = y[m_dep], y[m_arr]
        rows.append({
            "airport": ap_code,
            "n_dep": int(m_dep.sum()), "n_arr": int(m_arr.sum()),
            "mean_taxi_out": float(y_dep.mean()), "median_taxi_out": float(np.median(y_dep)),
            "mean_taxi_in": float(y_arr.mean()), "median_taxi_in": float(np.median(y_arr)),
            "p10_taxi_in": float(np.percentile(y_arr, 10)),
            "p90_taxi_in": float(np.percentile(y_arr, 90)),
            "mean_ratio_out_over_in": float(y_dep.mean() / y_arr.mean()),
            "median_ratio_out_over_in": float(np.median(y_dep) / np.median(y_arr)),
        })
    return sorted(rows, key=lambda r: -r["mean_ratio_out_over_in"])


def load_xy(featureset: str):
    X = D.build_scope("train2025", featureset)
    raw = cio.load_training(columns=[config.ID, "PHASE_mvt", config.TARGET])
    dep = raw[raw["PHASE_mvt"].eq("DEP")]
    y_by_id = pd.Series(dep[config.TARGET].astype("float64").to_numpy(),
                        index=dep[config.ID].to_numpy())
    ids = X[config.ID].to_numpy()
    y = y_by_id.reindex(ids).to_numpy()
    valid = np.isfinite(y) & (y > config.TARGET_IMPOSSIBLE_MAX_SEC)
    return X.loc[valid].reset_index(drop=True), y[valid]


def _rho(a: "np.ndarray", b: "np.ndarray") -> float:
    """Spearman rho between two numpy arrays, computed positionally.

    NOT `series_a.corr(series_b)` on columns pulled straight out of a
    DataFrame -- pandas aligns Series.corr by INDEX LABEL, and a boolean-mask
    slice keeps the original row labels while a freshly built `pd.Series(arr)`
    gets a fresh 0..n-1 label range. The two rarely match, so `.corr()` silently
    pairs the wrong rows together and drops the rest (verified: on a 4-row toy
    case it used 2 of 4 rows, and paired the WRONG values on both). Wrapping
    both sides fresh, from bare numpy arrays, forces position-based alignment.
    """
    return float(pd.Series(np.asarray(a)).corr(pd.Series(np.asarray(b)), method="spearman"))


def part_a(X: "pd.DataFrame", y: "np.ndarray") -> list:
    """Does taxi-out track inbound (arrival) traffic vs outbound (departure) traffic?"""
    rows = []
    airport = X[AIRPORT_COL].astype("object").to_numpy()
    cols = {c: X[c].to_numpy(dtype="float64") for c in
            ["arr_prev_15m", "arr_prev_30m", "arr_prev_60m",
             "dep_prev_15m", "dep_prev_30m", "dep_prev_60m", "same_rwy_prev_15m"]}
    for ap_code in sorted(pd.unique(airport)):
        m = airport == ap_code
        yy = y[m]
        row = {"airport": ap_code, "n": int(m.sum())}
        for w in (15, 30, 60):
            row["rho_arr_prev_{}m".format(w)] = _rho(cols["arr_prev_{}m".format(w)][m], yy)
            row["rho_dep_prev_{}m".format(w)] = _rho(cols["dep_prev_{}m".format(w)][m], yy)
        row["rho_same_rwy_prev_15m"] = _rho(cols["same_rwy_prev_15m"][m], yy)
        row["arr_minus_dep_15m"] = row["rho_arr_prev_15m"] - row["rho_dep_prev_15m"]
        rows.append(row)
    return sorted(rows, key=lambda r: -r["arr_minus_dep_15m"])


def _controlled_diff(y, is_biz, m_ap, group_col_values, min_n=20):
    """Business-minus-other taxi-out, within each level of group_col, n-weighted."""
    diffs, weights = [], []
    for level in pd.unique(group_col_values[m_ap]):
        if level is None or (isinstance(level, float) and np.isnan(level)):
            continue
        m_g = m_ap & (group_col_values == level)
        nb = int((m_g & is_biz).sum())
        no = int((m_g & ~is_biz).sum())
        if nb < min_n or no < min_n:
            continue
        diffs.append(float(y[m_g & is_biz].mean() - y[m_g & ~is_biz].mean()))
        weights.append(nb)
    if not diffs:
        return None, 0
    return float(np.average(diffs, weights=weights)), len(diffs)


def part_b(X: "pd.DataFrame", y: "np.ndarray") -> list:
    """Business segment taxi-out: raw, runway-controlled, and stand-controlled.

    v2 (advisor review after v1): v1 controlled only on RUNWAY_mvt. But
    features.py's own comment says stand->runway, not runway alone, is the
    strongest geometric determinant of taxi distance, and business jets
    typically sit on a different STAND set entirely (GA/FBO aprons) -- a
    runway-only control can leave that confound almost fully in place. EHAM's
    v1 result (raw -417s -> runway-controlled +56s, i.e. the "advantage" was
    ~entirely geometry) is direct evidence the geometry axis matters a lot;
    this version checks whether stand-level control does the same to the
    airports where the advantage "survived" runway control.
    """
    airport = X[AIRPORT_COL].astype("object").to_numpy()
    runway = X[RUNWAY_COL].astype("object").to_numpy()
    stand = X[STAND_COL].astype("object").to_numpy()
    seg = X[SEGMENT_COL].astype("object").to_numpy()
    is_biz = seg == "Business"

    rows = []
    for ap_code in sorted(pd.unique(airport)):
        m_ap = airport == ap_code
        if not (is_biz & m_ap).any():
            continue
        biz_mean = float(y[m_ap & is_biz].mean())
        other_mean = float(y[m_ap & ~is_biz].mean())
        n_biz = int((m_ap & is_biz).sum())
        n_other = int((m_ap & ~is_biz).sum())

        rwy_diff, n_rwy = _controlled_diff(y, is_biz, m_ap, runway)
        stand_diff, n_stand = _controlled_diff(y, is_biz, m_ap, stand)

        rows.append({
            "airport": ap_code, "n_business": n_biz, "n_other": n_other,
            "mean_taxiout_business": biz_mean, "mean_taxiout_other": other_mean,
            "raw_diff_business_minus_other": biz_mean - other_mean,
            "runway_controlled_diff": rwy_diff, "n_runways_used_for_control": n_rwy,
            "stand_controlled_diff": stand_diff, "n_stands_used_for_control": n_stand,
        })
    return sorted(rows, key=lambda r: r["raw_diff_business_minus_other"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    print("=" * 100)
    print("(a2) Taxi-OUT duration vs taxi-IN duration, per airport -- the direct")
    print("     operational reading of 'arrivals prioritized over departures'")
    print("=" * 100)
    a2_rows = part_a2()
    print("{:6s} {:>9s} {:>9s} {:>10s} {:>10s} {:>8s} {:>8s}".format(
        "adep", "n_dep", "n_arr", "mean_out", "mean_in", "ratio", "med_rat"))
    for r in a2_rows:
        print("{:6s} {:9d} {:9d} {:10.1f} {:10.1f} {:8.2f} {:8.2f}".format(
            r["airport"], r["n_dep"], r["n_arr"], r["mean_taxi_out"], r["mean_taxi_in"],
            r["mean_ratio_out_over_in"], r["median_ratio_out_over_in"]))
    print("\n  taxi-out longer than taxi-in (mean ratio > 1) at {}/{} airports".format(
        sum(1 for r in a2_rows if r["mean_ratio_out_over_in"] > 1), len(a2_rows)))

    X, y = load_xy(args.featureset)

    print("\n" + "=" * 100)
    print("(a) Arrival- vs departure-traffic correlation with taxi-out, per airport")
    print("=" * 100)
    a_rows = part_a(X, y)
    print("{:6s} {:>8s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "adep", "n", "rho_arr15", "rho_dep15", "rho_rwy15", "arr-dep15", "rho_arr60"))
    for r in a_rows:
        print("{:6s} {:8d} {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}".format(
            r["airport"], r["n"], r["rho_arr_prev_15m"], r["rho_dep_prev_15m"],
            r["rho_same_rwy_prev_15m"], r["arr_minus_dep_15m"], r["rho_arr_prev_60m"]))
    n_arr_wins = sum(1 for r in a_rows if r["arr_minus_dep_15m"] > 0)
    print("\n  arrival-traffic rho > departure-traffic rho at {}/{} airports".format(
        n_arr_wins, len(a_rows)))

    print("\n" + "=" * 100)
    print("(b) Business-segment taxi-out vs other segments, per airport")
    print("=" * 100)
    b_rows = part_b(X, y)
    print("{:6s} {:>9s} {:>9s} {:>10s} {:>10s} {:>10s}".format(
        "adep", "n_biz", "n_other", "raw_diff", "rwy_ctrl", "stand_ctrl"))
    for r in b_rows:
        print("{:6s} {:9d} {:9d} {:10.1f} {:>10s} {:>10s}".format(
            r["airport"], r["n_business"], r["n_other"],
            r["raw_diff_business_minus_other"],
            "{:.1f}".format(r["runway_controlled_diff"])
                if r["runway_controlled_diff"] is not None else "n/a",
            "{:.1f}".format(r["stand_controlled_diff"])
                if r["stand_controlled_diff"] is not None else "n/a"))
    n_neg_raw = sum(1 for r in b_rows if r["raw_diff_business_minus_other"] < 0)
    rwy_ctrl = [r["runway_controlled_diff"] for r in b_rows if r["runway_controlled_diff"] is not None]
    stand_ctrl = [r["stand_controlled_diff"] for r in b_rows if r["stand_controlled_diff"] is not None]
    print("\n  business faster (raw) at {}/{} airports".format(n_neg_raw, len(b_rows)))
    print("  faster (runway-controlled) at {}/{}".format(
        sum(1 for d in rwy_ctrl if d < 0), len(rwy_ctrl)))
    print("  faster (stand-controlled) at {}/{}".format(
        sum(1 for d in stand_ctrl if d < 0), len(stand_ctrl)))

    report = {"featureset": args.featureset,
              "part_a2_taxi_out_vs_taxi_in_duration": a2_rows,
              "part_a_arrival_vs_departure_traffic_correlation": a_rows,
              "part_b_business_segment": b_rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_guarded(OUT_DIR / "priority_rules.json", report, force=args.force)
    print("\nwrote {}".format(OUT_DIR / "priority_rules.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
