"""Feature construction. Every feature here is derivable from the columns that
survive into ranking.parquet's DEP rows -- enforced by `assert_ranking_computable`.

Two design rules govern this module:

1. **Causal congestion.** Airport congestion is the dominant driver of taxi-out
   variance, but the obvious encoding (rolling mean of recent taxi-out times) is
   unavailable at scoring time: taxi-out is exactly what ranking.parquet blanks.
   So congestion is built only from movement TIMES (takeoff/landing instants,
   given for every row) and from ARRIVAL taxi-in times (not blanked). Counting
   movements in a window AROUND the flight -- including after it -- is legitimate
   here: the ranking file ships every movement in the scored months, so the
   forward window is as observable at scoring time as the backward one.

2. **The proxy is a switch, not a hardcode.** `AOBT_3_flt` (actual off-block from
   the flown trajectory) is a near-target proxy the organisers left in the scored
   set. `featureset="no_aobt"` removes it and every feature derived from it, so
   the ablation is one flag rather than a second copy of this file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

CATEGORICAL = [
    "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "ADES_mvt",
    "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt",
    "stand_runway", "adep_runway", "adep_stand", "dest_region",
    "adep_flight_record",
]

WINDOWS_MIN = (15, 30, 60)


# --------------------------------------------------------------------------- #
# congestion
# --------------------------------------------------------------------------- #
def _counts_in_window(event_times_ns, query_times_ns, lo_ns, hi_ns):
    """For each query time q, count events in [q+lo, q+hi). Both arrays in ns.

    `event_times_ns` must be sorted. Uses searchsorted, so this is O(n log n)
    rather than the O(n^2) a groupby-apply over 4.2M rows would cost.
    """
    left = np.searchsorted(event_times_ns, query_times_ns + lo_ns, side="left")
    right = np.searchsorted(event_times_ns, query_times_ns + hi_ns, side="left")
    return (right - left).astype("int32")


def _rolling_prev_mean(event_times_ns, event_values, query_times_ns, window_ns):
    """Mean of event_values over events in [q-window, q). NaN when the window is empty.

    Prefix sums over the sorted event array; strictly backward-looking, so it is
    safe to build from arrival taxi-in times even though those are outcomes.
    """
    vals = np.asarray(event_values, dtype="float64")
    ok = np.isfinite(vals)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, vals, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok.astype("int64"))])
    left = np.searchsorted(event_times_ns, query_times_ns - window_ns, side="left")
    right = np.searchsorted(event_times_ns, query_times_ns, side="left")
    n = ccnt[right] - ccnt[left]
    s = csum[right] - csum[left]
    out = np.full(query_times_ns.shape, np.nan, dtype="float64")
    nz = n > 0
    out[nz] = s[nz] / n[nz]
    return out


def build_congestion(mov: "pd.DataFrame", featureset: str) -> "pd.DataFrame":
    """Per-airport traffic-density features for every DEP row in `mov`.

    `mov` must contain BOTH phases for the airports/period being featurised --
    arrivals are what make the runway-crossing and apron-occupancy signals
    visible. Returns a frame indexed like `mov`'s DEP rows.

    Accumulates into positional numpy arrays and assigns once at the end: a
    per-airport, per-runway `.loc` write against a 2M-row index is the slow,
    alignment-dependent way to do the same thing.
    """
    need = {"ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
            "RUNWAY_mvt", config.TARGET}
    missing = need - set(mov.columns)
    if missing:
        raise KeyError("build_congestion needs {}".format(sorted(missing)))

    is_dep = mov["PHASE_mvt"].eq("DEP").to_numpy()
    # The observed airport: ADEP for a departure, ADES for an arrival.
    airport = np.where(is_dep, mov["ADEP_mvt"].to_numpy(), mov["ADES_mvt"].to_numpy())
    t_ns = mov["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()
    target = mov[config.TARGET].to_numpy(dtype="float64", na_value=np.nan)

    dep_idx = np.flatnonzero(is_dep)
    n = dep_idx.size
    # position of each global row within dep_idx, for scatter-writes
    pos_of_row = np.full(len(mov), -1, dtype="int64")
    pos_of_row[dep_idx] = np.arange(n)

    cols = {}
    for w in WINDOWS_MIN:
        cols["dep_prev_{}m".format(w)] = np.zeros(n, dtype="int32")
        cols["dep_next_{}m".format(w)] = np.zeros(n, dtype="int32")
        cols["arr_prev_{}m".format(w)] = np.zeros(n, dtype="int32")
    cols["same_rwy_prev_15m"] = np.zeros(n, dtype="int32")
    cols["same_rwy_prev_30m"] = np.zeros(n, dtype="int32")
    cols["same_rwy_next_15m"] = np.zeros(n, dtype="int32")
    cols["same_rwy_next_30m"] = np.zeros(n, dtype="int32")
    cols["arr_taxiin_mean_prev_60m"] = np.full(n, np.nan, dtype="float64")
    # Design-gate finding S3: apron occupancy must exist in BOTH featuresets, or
    # "what is AOBT worth" silently also measures "what is apron congestion
    # worth". NaN (not 0) where no off-block estimate exists -- 0 would mean
    # "empty apron", which is a different claim from "unknown".
    cols["n_taxiing_at_pushback"] = np.full(n, np.nan, dtype="float64")

    minute = 60 * 1_000_000_000
    rwy_all = mov["RUNWAY_mvt"].to_numpy()

    # The off-block instant each departure pushed back, by the best estimate this
    # featureset is allowed. with_aobt reads the measured AOBT_3; no_aobt falls
    # back to the flight plan's estimated off-block (EOBT_1), which is a biased
    # but usable anchor -- and a COUNT is far more robust to a common bias than a
    # direct proxy is, because a shared offset shifts every aircraft equally.
    if featureset == "with_aobt" and "AOBT_3_flt" in mov.columns:
        ob_series = mov["AOBT_3_flt"]
    elif "EOBT_1_flt" in mov.columns:
        ob_series = mov["EOBT_1_flt"]
    else:
        ob_series = None
    if ob_series is not None:
        aobt_ns = ob_series.astype("int64").to_numpy()
        aobt_ok = ob_series.notna().to_numpy()
        # an estimate that lands after takeoff is not an off-block time
        aobt_ok = aobt_ok & (aobt_ns < t_ns)

    for ap in pd.unique(airport):
        if ap is None or (isinstance(ap, float) and np.isnan(ap)):
            continue
        at_ap = airport == ap
        ap_dep = np.flatnonzero(at_ap & is_dep)
        if ap_dep.size == 0:
            continue
        ap_arr = np.flatnonzero(at_ap & ~is_dep)
        p = pos_of_row[ap_dep]

        q = t_ns[ap_dep]                       # query times: this airport's takeoffs
        dep_sorted = np.sort(t_ns[ap_dep])
        arr_order = np.argsort(t_ns[ap_arr], kind="stable")
        arr_sorted = t_ns[ap_arr][arr_order]

        for w in WINDOWS_MIN:
            wn = w * minute
            # half-open [q-w, q) excludes the flight itself from its own count
            cols["dep_prev_{}m".format(w)][p] = _counts_in_window(dep_sorted, q, -wn, 0)
            cols["dep_next_{}m".format(w)][p] = _counts_in_window(dep_sorted, q, 1, wn)
            cols["arr_prev_{}m".format(w)][p] = _counts_in_window(arr_sorted, q, -wn, 0)

        if arr_sorted.size:
            cols["arr_taxiin_mean_prev_60m"][p] = _rolling_prev_mean(
                arr_sorted, target[ap_arr][arr_order], q, 60 * minute)

        # queue on the SAME runway -- the direct departure-queue signal
        rwy_here = rwy_all[ap_dep]
        for r in pd.unique(rwy_here):
            if r is None or (isinstance(r, float) and np.isnan(r)):
                continue
            sel = rwy_here == r
            tt = q[sel]
            srt = np.sort(tt)
            cols["same_rwy_prev_15m"][p[sel]] = _counts_in_window(
                srt, tt, -15 * minute, 0)
            cols["same_rwy_prev_30m"][p[sel]] = _counts_in_window(
                srt, tt, -30 * minute, 0)
            cols["same_rwy_next_15m"][p[sel]] = _counts_in_window(
                srt, tt, 1, 15 * minute)
            cols["same_rwy_next_30m"][p[sel]] = _counts_in_window(
                srt, tt, 1, 30 * minute)

        # apron occupancy: how many aircraft were between off-block and takeoff
        # at the moment this one pushed back.
        if ob_series is not None:
            ok = aobt_ok[ap_dep]
            if ok.any():
                qb = aobt_ns[ap_dep][ok]
                ob_sorted = np.sort(qb)
                to_sorted = np.sort(q[ok])
                started = np.searchsorted(ob_sorted, qb, side="left")
                finished = np.searchsorted(to_sorted, qb, side="left")
                cols["n_taxiing_at_pushback"][p[ok]] = (
                    started - finished).astype("float64")

    cols["dep_surge_ratio_15_60"] = ((cols["dep_prev_15m"] + 1.0) /
                                     (cols["dep_prev_60m"] / 4.0 + 1.0)).astype("float64")
    return pd.DataFrame(cols, index=mov.index[dep_idx])


# --------------------------------------------------------------------------- #
# main builder
# --------------------------------------------------------------------------- #
ABSOLUTE_TIME_COLS = ["month_num", "doy", "doy_sin", "doy_cos"]


def build_features(mov: "pd.DataFrame", featureset: str = config.DEFAULT_FEATURESET
                   ) -> "pd.DataFrame":
    """Featurise the DEP rows of `mov`. Returns a frame indexed like those rows.

    Note on ABSOLUTE_TIME_COLS (design-gate finding S10): under the `forward`
    split the model trains on Jan-Aug and is tested on Nov-Dec, so `month_num`
    and `doy` are out of the training range on every test row. A GBM puts them in
    August's rightmost bin; the ridge extrapolates smoothly through the sin/cos
    pair. That is a split artifact sitting directly on the axis the arms are
    ranked on, and it disappears in the real task (the 12-month refit sees every
    month). They are kept -- they carry real seasonal signal -- but the arm
    ranking must be confirmed with them dropped; see stage5_ablations.py.
    """
    if featureset not in config.FEATURESETS:
        raise ValueError("featureset must be one of {}, got {!r}".format(
            config.FEATURESETS, featureset))

    dep = mov[mov["PHASE_mvt"].eq("DEP")]
    t = dep["MVT_TIME_UTC_mvt"]
    X = pd.DataFrame(index=dep.index)

    # ---- temporal (from the takeoff instant, which ranking.parquet ships) ----
    X["hour"] = t.dt.hour.astype("int16")
    X["minute_of_day"] = (t.dt.hour * 60 + t.dt.minute).astype("int16")
    X["dow"] = t.dt.dayofweek.astype("int8")
    X["month_num"] = t.dt.month.astype("int8")
    X["doy"] = t.dt.dayofyear.astype("int16")
    X["is_weekend"] = (t.dt.dayofweek >= 5).astype("int8")
    ang = 2 * np.pi * X["minute_of_day"] / 1440.0
    X["tod_sin"], X["tod_cos"] = np.sin(ang), np.cos(ang)
    yang = 2 * np.pi * X["doy"] / 365.25
    X["doy_sin"], X["doy_cos"] = np.sin(yang), np.cos(yang)

    # ---- schedule deltas: how late is this flight relative to each plan? ----
    for col, name in (("SCHED_TIME_UTC_mvt", "sched"), ("EOBT_1_flt", "eobt"),
                      ("IOBT_flt", "iobt"), ("LOBT_flt", "lobt")):
        X["delta_{}_sec".format(name)] = (t - dep[col]).dt.total_seconds()
    # plan churn: how much the off-block estimate moved between filings
    X["eobt_minus_iobt_sec"] = (dep["EOBT_1_flt"] - dep["IOBT_flt"]).dt.total_seconds()
    X["lobt_minus_iobt_sec"] = (dep["LOBT_flt"] - dep["IOBT_flt"]).dt.total_seconds()

    # ---- the near-target proxy, isolated behind the featureset switch ----
    if featureset == "with_aobt":
        X["delta_aobt_sec"] = (t - dep["AOBT_3_flt"]).dt.total_seconds()
        X["aobt_minus_eobt_sec"] = (
            dep["AOBT_3_flt"] - dep["EOBT_1_flt"]).dt.total_seconds()
        X["aobt_minus_sched_sec"] = (
            dep["AOBT_3_flt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
        X["aobt_minus_lobt_sec"] = (
            dep["AOBT_3_flt"] - dep["LOBT_flt"]).dt.total_seconds()
        X["has_aobt"] = dep["AOBT_3_flt"].notna().astype("int8")

    # ---- flags ----
    X["has_flight_record"] = dep["AIRCRAFT_OPERATOR_flt"].notna().astype("int8")
    X["diverted_plan"] = (dep["ADES_flt"] != dep["ADES_FILED_flt"]).astype("int8")
    X["mvt_flt_type_disagree"] = (
        dep["AIRCRAFT_TYPE_mvt"] != dep["AIRCRAFT_TYPE_flt"]).astype("int8")
    X["abs_delta_sched_sec"] = X["delta_sched_sec"].abs()
    X["abs_delta_eobt_sec"] = X["delta_eobt_sec"].abs()

    # ---- categoricals, incl. the geometry pair that sets taxi distance ----
    adep = dep["ADEP_mvt"].astype("string")
    stand = dep["STAND_mvt"].astype("string")
    rwy = dep["RUNWAY_mvt"].astype("string")
    X["ADEP_mvt"] = adep
    X["RUNWAY_mvt"] = rwy
    X["STAND_mvt"] = stand
    X["ADES_mvt"] = dep["ADES_mvt"].astype("string")
    X["AIRCRAFT_TYPE_mvt"] = dep["AIRCRAFT_TYPE_mvt"].astype("string")
    X["AIRCRAFT_OPERATOR_flt"] = dep["AIRCRAFT_OPERATOR_flt"].astype("string")
    X["WK_TBL_CAT_flt"] = dep["WK_TBL_CAT_flt"].astype("string")
    X["MARKET_SEGMENT_flt"] = dep["MARKET_SEGMENT_flt"].astype("string")
    X["FLIGHT_TYPE_flt"] = dep["FLIGHT_TYPE_flt"].astype("string")
    X["FLIGHT_RULE_mvt"] = dep["FLIGHT_RULE_mvt"].astype("string")
    # stand->runway is the single strongest geometric determinant of taxi distance
    X["adep_stand"] = adep + "|" + stand.fillna("NA")
    X["adep_runway"] = adep + "|" + rwy.fillna("NA")
    X["stand_runway"] = adep + "|" + stand.fillna("NA") + "|" + rwy.fillna("NA")
    X["adep_flight_record"] = adep + "|" + X["has_flight_record"].astype("string")
    # ICAO prefix as a coarse destination region (no coordinates are shipped)
    X["dest_region"] = dep["ADES_mvt"].astype("string").str.slice(0, 2)

    # ---- congestion ----
    cong = build_congestion(mov, featureset)
    X = X.join(cong, how="left")

    for c in CATEGORICAL:
        X[c] = X[c].astype("category")

    return X


def source_columns(featureset: str) -> "list[str]":
    """The raw columns build_features reads -- the argument to the ranking gate."""
    base = ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt",
            "LOBT_flt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "ADES_flt",
            "ADES_FILED_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
            "AIRCRAFT_TYPE_flt", "AIRCRAFT_OPERATOR_flt", "WK_TBL_CAT_flt",
            "MARKET_SEGMENT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]
    if featureset == "with_aobt":
        base += list(config.PROXY_COLS)
    return base
