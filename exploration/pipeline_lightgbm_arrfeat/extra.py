"""Extra features from the two channels the base feature set under-uses.

Why these two, and why they are new information rather than a re-slice
---------------------------------------------------------------------
After `cross_blend`, **82.8% of the remaining test error is the 327,948 in-band
rows** (in-band RMSE 224.27), not the tail. Bulk accuracy is now the constraint,
and the only way to move it is to give the model something it does not already
have. Two such channels exist and are barely touched:

**1. Arrival taxi-in times — observed ground truth, available at prediction
time.** `config.BLANKED_ON_RANKING_DEP` blanks `BLOCK_TIME_UTC_mvt` and the
target on **DEP** rows only. Arrival rows keep theirs, and the audit confirms
they are present and undrifted in the scored file (`arr_taxiin` train_mean
536.71 vs rank_mean 534.98). So for any departure we may look at how long
aircraft that *landed* in the previous hour took to taxi in — a direct,
measured reading of how the airfield is running right now, not a proxy for it.
The base feature set spends exactly one column on this
(`arr_taxiin_mean_prev_60m`). This module adds the window, dispersion and
per-runway structure it leaves on the table. Dispersion is the interesting one:
a high *variance* of taxi-in times is the signature of a disrupted airfield in a
way a mean is not.

**2. The queue that actually matters is runway-specific.**
`n_taxiing_at_pushback` counts every aircraft between off-block and takeoff
anywhere on the airport. But a departure queues behind the aircraft heading for
*its* runway; traffic bound for the other side of the field is not in front of
it. `n_taxiing_same_rwy_at_pushback` restricts the same construction to the
runway this flight will use.

Leakage discipline
------------------
Every column here is a function of *other* movements' timestamps, plus arrival
targets which are not blanked in the scored set. Nothing reads this row's own
target, and nothing reads any DEP target — so this is not target encoding and
there is no in-fold leak surface. Windows are half-open and backward-looking
except where a column's name says `next`, matching `common/features.py`.

Kept OUT of `common/features.py` deliberately: adding columns there changes
`config.feature_code_fingerprint()`, which invalidates the feature cache and
every run artifact in the build. This module is the experiment's own scope, so
the ten existing arms stay comparable while it is measured.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import io as cio

MINUTE_NS = 60 * 1_000_000_000


def code_fingerprint() -> str:
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    return h.hexdigest()[:16]


def _prev_stats(event_ns, values, query_ns, window_ns):
    """mean / std / max of `values` for events in [q - window, q).

    One sorted pass with prefix sums for mean and std; the max needs a per-query
    slice, so it is computed only where the window is non-empty.
    """
    order = np.argsort(event_ns, kind="stable")
    e, v = event_ns[order], np.asarray(values, dtype="float64")[order]
    ok = np.isfinite(v)
    e, v = e[ok], v[ok]
    n = e.size
    out_mean = np.full(query_ns.size, np.nan)
    out_std = np.full(query_ns.size, np.nan)
    out_max = np.full(query_ns.size, np.nan)
    if n == 0:
        return out_mean, out_std, out_max
    cs = np.concatenate([[0.0], np.cumsum(v)])
    cs2 = np.concatenate([[0.0], np.cumsum(v * v)])
    lo = np.searchsorted(e, query_ns - window_ns, side="left")
    hi = np.searchsorted(e, query_ns, side="left")
    cnt = (hi - lo).astype("float64")
    nz = cnt > 0
    s = cs[hi] - cs[lo]
    s2 = cs2[hi] - cs2[lo]
    out_mean[nz] = s[nz] / cnt[nz]
    var = np.zeros_like(s)
    var[nz] = np.maximum(s2[nz] / cnt[nz] - out_mean[nz] ** 2, 0.0)
    out_std[nz] = np.sqrt(var[nz])
    for i in np.flatnonzero(nz):
        out_max[i] = v[lo[i]:hi[i]].max()
    return out_mean, out_std, out_max


def build_extra(mov: "pd.DataFrame", featureset: str) -> "pd.DataFrame":
    """Extra columns, indexed like `mov`'s DEP rows (same contract as build_congestion)."""
    need = {"ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
            "RUNWAY_mvt", config.TARGET}
    missing = need - set(mov.columns)
    if missing:
        raise KeyError("build_extra needs {}".format(sorted(missing)))

    is_dep = mov["PHASE_mvt"].eq("DEP").to_numpy()
    airport = np.where(is_dep, mov["ADEP_mvt"].to_numpy(), mov["ADES_mvt"].to_numpy())
    t_ns = mov["MVT_TIME_UTC_mvt"].astype("int64").to_numpy()
    target = mov[config.TARGET].to_numpy(dtype="float64", na_value=np.nan)
    rwy_all = mov["RUNWAY_mvt"].to_numpy()

    dep_idx = np.flatnonzero(is_dep)
    n = dep_idx.size
    pos_of_row = np.full(len(mov), -1, dtype="int64")
    pos_of_row[dep_idx] = np.arange(n)

    cols = {c: np.full(n, np.nan, dtype="float64") for c in (
        "arr_taxiin_mean_prev_30m", "arr_taxiin_mean_prev_120m",
        "arr_taxiin_std_prev_60m", "arr_taxiin_max_prev_60m",
        "arr_taxiin_mean_prev_60m_same_rwy", "arr_taxiin_std_prev_60m_same_rwy",
        "n_taxiing_same_rwy_at_pushback")}

    if featureset == "with_aobt" and "AOBT_3_flt" in mov.columns:
        ob = mov["AOBT_3_flt"]
    elif "EOBT_1_flt" in mov.columns:
        ob = mov["EOBT_1_flt"]
    else:
        ob = None
    if ob is not None:
        ob_ns = ob.astype("int64").to_numpy()
        ob_ok = ob.notna().to_numpy() & (ob_ns < t_ns)

    for ap in pd.unique(airport):
        if ap is None or (isinstance(ap, float) and np.isnan(ap)):
            continue
        at = airport == ap
        ap_dep = np.flatnonzero(at & is_dep)
        if ap_dep.size == 0:
            continue
        ap_arr = np.flatnonzero(at & ~is_dep)
        p = pos_of_row[ap_dep]
        q = t_ns[ap_dep]

        if ap_arr.size:
            ae, av = t_ns[ap_arr], target[ap_arr]
            m30, _, _ = _prev_stats(ae, av, q, 30 * MINUTE_NS)
            m120, _, _ = _prev_stats(ae, av, q, 120 * MINUTE_NS)
            _, s60, x60 = _prev_stats(ae, av, q, 60 * MINUTE_NS)
            cols["arr_taxiin_mean_prev_30m"][p] = m30
            cols["arr_taxiin_mean_prev_120m"][p] = m120
            cols["arr_taxiin_std_prev_60m"][p] = s60
            cols["arr_taxiin_max_prev_60m"][p] = x60

        # ---- per-runway structure ----
        rwy_dep = rwy_all[ap_dep]
        rwy_arr = rwy_all[ap_arr] if ap_arr.size else np.array([])
        for r in pd.unique(rwy_dep):
            if r is None or (isinstance(r, float) and np.isnan(r)):
                continue
            sel = rwy_dep == r
            qq = q[sel]
            if ap_arr.size:
                asel = rwy_arr == r
                if asel.any():
                    mr, sr, _ = _prev_stats(t_ns[ap_arr][asel], target[ap_arr][asel],
                                            qq, 60 * MINUTE_NS)
                    cols["arr_taxiin_mean_prev_60m_same_rwy"][p[sel]] = mr
                    cols["arr_taxiin_std_prev_60m_same_rwy"][p[sel]] = sr
            # the queue that is actually ahead of this flight: aircraft bound for
            # THIS runway that had pushed back but not yet taken off
            if ob is not None:
                ok = ob_ok[ap_dep] & sel
                if ok.any():
                    qb = ob_ns[ap_dep][ok]
                    started = np.searchsorted(np.sort(ob_ns[ap_dep][ok]), qb, side="left")
                    finished = np.searchsorted(np.sort(q[ok]), qb, side="left")
                    cols["n_taxiing_same_rwy_at_pushback"][p[ok]] = (
                        started - finished).astype("float64")

    return pd.DataFrame(cols, index=mov.index[dep_idx])


ADDED_COLUMNS = ["arr_taxiin_mean_prev_30m", "arr_taxiin_mean_prev_120m",
                 "arr_taxiin_std_prev_60m", "arr_taxiin_max_prev_60m",
                 "arr_taxiin_mean_prev_60m_same_rwy",
                 "arr_taxiin_std_prev_60m_same_rwy",
                 "n_taxiing_same_rwy_at_pushback",
                 # per-STAND arrival taxi-in: the distance proxy (see below)
                 "stand_taxiin_mean", "stand_taxiin_p10", "stand_taxiin_n",
                 "stand_rwy_taxiin_mean"]


def add_stand_taxiin(mov: "pd.DataFrame", out: "pd.DataFrame") -> "pd.DataFrame":
    """Per-STAND arrival taxi-in statistics -- a learned proxy for stand geometry.

    The idea, and the measurement behind it
    ---------------------------------------
    How long an aircraft takes to taxi IN to a stand and how long it takes to taxi
    OUT from it are both functions of the same thing: how far that stand is from
    the runway system. The release ships no coordinates, but it does ship arrival
    taxi-in times -- on 100% of arrival rows, with a stand on 100% of them, and
    **not blanked in the scored set**. So the geometry is observable after all.

    Measured across 2,158 stands with >=30 movements each way, the correlation
    between a stand's mean taxi-out and its mean taxi-in is 0.445 overall
    (Spearman 0.502) -- but the aggregate badly understates it, because the
    relationship is **airport-specific**:

        EHAM  0.904 | LEMD 0.788 | LIRF 0.617 | LFPG 0.587 | EDDF 0.489
        LTFM  0.172 | EDDM 0.160 | LSZH 0.125 | LEBL **-0.244**

    At EHAM a stand's taxi-in nearly determines its taxi-out. At LEBL the sign
    *inverts* -- consistent with arrivals and departures using different runways,
    so a stand near the landing runway is far from the departure runway. A pooled
    model would average 0.90 and -0.24 into noise; the per-airport arms can use
    each airport's own sign and strength, which is why this feature belongs with
    that family.

    `stand_taxiin_p10` is the low-quantile version -- closer to an *unimpeded*
    taxi-in, so a cleaner distance proxy than the mean, which carries arrival
    congestion. `stand_taxiin_n` is the sample size behind the estimate, so a
    tree can discount a stand seen five times.

    Computed scope-locally (train statistics for training rows, 2026 statistics
    for scored rows), matching how the congestion features already work: it
    describes the period being predicted, and uses no departure target.
    """
    if "STAND_mvt" not in mov.columns:
        raise KeyError("add_stand_taxiin needs STAND_mvt; loaded columns are {}. "
                       "This used to fail only AFTER build_extra had run."
                       .format(sorted(mov.columns)))
    is_dep = mov["PHASE_mvt"].eq("DEP")
    arr = mov[~is_dep]
    if arr.empty:
        for c in ("stand_taxiin_mean", "stand_taxiin_p10", "stand_taxiin_n",
                  "stand_rwy_taxiin_mean"):
            out[c] = np.nan
        return out

    ap_arr = arr["ADES_mvt"].astype(str)
    key_a = ap_arr + "|" + arr["STAND_mvt"].astype(str)
    t = arr[config.TARGET].astype("float64")
    g = t.groupby(key_a, observed=True)
    stats = pd.DataFrame({"stand_taxiin_mean": g.mean(),
                          "stand_taxiin_p10": g.quantile(0.10),
                          "stand_taxiin_n": g.size()})
    # arrivals that used the same runway, for airports where the pairing matters
    key_ar = key_a + "|" + arr["RUNWAY_mvt"].astype(str)
    stats_r = t.groupby(key_ar, observed=True).mean().rename("stand_rwy_taxiin_mean")

    dep = mov[is_dep]
    key_d = dep["ADEP_mvt"].astype(str) + "|" + dep["STAND_mvt"].astype(str)
    for c in ("stand_taxiin_mean", "stand_taxiin_p10", "stand_taxiin_n"):
        out[c] = key_d.map(stats[c]).to_numpy(dtype="float64")
    key_dr = key_d + "|" + dep["RUNWAY_mvt"].astype(str)
    out["stand_rwy_taxiin_mean"] = key_dr.map(stats_r).to_numpy(dtype="float64")
    return out


def build_scope_extra(scope: str, featureset: str, *, force: bool = False):
    """Extra features for a whole scope, cached and keyed on THIS file's hash.

    Same S6 discipline as the base cache: without the code fingerprint in the
    path, editing this module would leave a stale cache in place and every run
    would silently train on the old columns.
    """
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FEATURE_DIR / "extra__{}__{}__{}.parquet".format(
        scope, featureset, code_fingerprint())
    if path.exists() and not force:
        return pd.read_parquet(path)
    # STAND_mvt is needed by add_stand_taxiin(); it was absent from this list when
    # that function was added, which cost a full feature build before the KeyError.
    cols = sorted({"MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
                   "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "STAND_mvt", config.TARGET,
                   "AOBT_3_flt", "EOBT_1_flt"})
    mov = cio.load_training(columns=cols) if scope == "train2025" else cio.load_ranking(columns=cols)
    X = build_extra(mov, featureset)
    X = add_stand_taxiin(mov, X)
    X[config.ID] = mov.loc[X.index, config.ID].to_numpy()
    X = X.set_index(config.ID)
    X.to_parquet(path)
    print("  [arrfeat] built {} extra columns for {} -> {}".format(
        len(ADDED_COLUMNS), scope, path.name))
    return X
