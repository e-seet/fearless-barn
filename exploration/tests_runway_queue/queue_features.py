#!/usr/bin/env python3
"""Runway queue / runway mode features from the movement records (PREREGISTERED.md Strategy 1), for every 2025
and 2026 departure. Only movements strictly before the departure's takeoff are used. Cached to queue.parquet."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from common import io as cio

QF = ["rq_dep15", "rq_dep30", "rq_gap_prev", "rq_arr15", "rm_dep_rw", "rm_arr_rw", "rm_mixed"]
CACHE = HERE / "queue.parquet"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def build():
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"]
    m = pd.concat([cio.load_training(columns=cols), cio.load_ranking(columns=cols)]).drop_duplicates(["MVT_ID_mvt", "PHASE_mvt"])
    m["t"] = (pd.to_datetime(m.MVT_TIME_UTC_mvt, utc=True) - EPOCH).dt.total_seconds()
    m["ap"] = np.where(m.PHASE_mvt.eq("DEP"), m.ADEP_mvt, m.ADES_mvt)
    m["rw"] = m.RUNWAY_mvt.astype(str)
    m = m[np.isfinite(m.t)]
    out = []
    for ap, g in m.groupby("ap"):
        dep = g[g.PHASE_mvt.eq("DEP")].sort_values("t"); arr = g[g.PHASE_mvt.eq("ARR")].sort_values("t")
        if dep.empty:
            continue
        res = pd.DataFrame(index=dep.MVT_ID_mvt.astype("int64").to_numpy())
        T = dep.t.to_numpy(); R = dep.rw.to_numpy()
        for c in QF:
            res[c] = np.nan
        # per-runway counts and gaps (departures and arrivals strictly before T)
        for rw in np.unique(R):
            md = R == rw
            td = T[md]                                   # sorted (dep sorted by t)
            ta = np.sort(arr.t.to_numpy()[arr.rw.to_numpy() == rw])
            lo15 = np.searchsorted(td, td - 900, "left"); lo30 = np.searchsorted(td, td - 1800, "left"); hi = np.searchsorted(td, td, "left")
            res.loc[res.index[md], "rq_dep15"] = hi - lo15
            res.loc[res.index[md], "rq_dep30"] = hi - lo30
            prev = np.where(hi > 0, td - td[np.maximum(hi - 1, 0)], np.nan)
            res.loc[res.index[md], "rq_gap_prev"] = np.minimum(prev, 3600)
            res.loc[res.index[md], "rq_arr15"] = np.searchsorted(ta, td, "left") - np.searchsorted(ta, td - 900, "left")
            res.loc[res.index[md], "rm_mixed"] = ((np.searchsorted(ta, td, "left") - np.searchsorted(ta, td - 3600, "left")) > 0).astype(float)
        # runway mode: distinct runways in the previous hour (O(n * k) over a small k of runways)
        for kind, src, col in (("dep", dep, "rm_dep_rw"), ("arr", arr, "rm_arr_rw")):
            cnt = np.zeros(len(T))
            for rw in src.rw.unique():
                ts = np.sort(src.t.to_numpy()[src.rw.to_numpy() == rw])
                cnt += (np.searchsorted(ts, T, "left") - np.searchsorted(ts, T - 3600, "left")) > 0
            res[col] = cnt
        out.append(res)
    q = pd.concat(out); q = q[~q.index.duplicated()]
    q.to_parquet(CACHE)
    return q


def load():
    return pd.read_parquet(CACHE) if CACHE.exists() else build()


if __name__ == "__main__":
    q = build()
    ap = pd.concat([cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"]), cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"])])
    ap = ap[ap.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index(ap[ap.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").MVT_ID_mvt.astype("int64")).ADEP_mvt
    print(len(q), "departures"); print(q.groupby(ap.reindex(q.index).to_numpy())[QF].mean().round(2).to_string())
