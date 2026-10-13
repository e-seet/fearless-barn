#!/usr/bin/env python3
"""Traffic features from other departures' ADS-B tracks (PREREGISTERED_v48.md), for every 2025 training and
2026 ranking departure. Cached to traffic.parquet (index MVT_ID_mvt)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from attach import estimates
from common import io as cio

TRAFFIC = ["tr_n30", "tr_stop30", "tr_lineup30", "tr_taxi30", "tr_gs30", "tr_ground_T", "tr_ground_T10"]
CACHE = HERE / "traffic.parquet"


def build() -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]
    d = pd.concat([cio.load_training(columns=cols), cio.load_ranking(columns=cols)])
    d = d[d.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt")
    d["T"] = (pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True) - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    d = d[np.isfinite(d["T"])].set_index(d.MVT_ID_mvt.astype("int64"))
    e = estimates()
    d = d.join(e[["e_move", "stop_s", "lineup_s", "med_gs"]])
    out = []
    for ap, g in d.groupby("ADEP_mvt"):
        g = g.sort_values("T")
        t = g["T"].to_numpy()
        matched = g.e_move.notna().to_numpy()
        res = pd.DataFrame(index=g.index)
        lo = np.searchsorted(t, t - 1800, "left"); hi = np.searchsorted(t, t, "left")
        cm = np.concatenate([[0], np.cumsum(matched)])
        res["tr_n30"] = cm[hi] - cm[lo]
        for col, name in (("stop_s", "tr_stop30"), ("lineup_s", "tr_lineup30"), ("e_move", "tr_taxi30"), ("med_gs", "tr_gs30")):
            v = g[col].to_numpy(dtype="float64"); ok = np.isfinite(v)
            cs = np.concatenate([[0], np.cumsum(np.where(ok, v, 0.0))]); cn = np.concatenate([[0], np.cumsum(ok)])
            n = cn[hi] - cn[lo]
            res[name] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        em = g.e_move.to_numpy(dtype="float64"); ok = np.isfinite(em)
        starts = np.sort((t - em)[ok]); ends = np.sort(t[ok])
        for x_off, name in ((0, "tr_ground_T"), (600, "tr_ground_T10")):
            x = t - x_off
            n = np.searchsorted(starts, x, "right") - np.searchsorted(ends, x, "left")
            self_in = ok & (np.where(ok, em, -1) >= x_off)          # own aircraft counted when it was already moving at x
            res[name] = n - self_in.astype(int)
        res["ap"] = ap
        out.append(res)
    tr = pd.concat(out)
    tr.to_parquet(CACHE)
    return tr


def load() -> pd.DataFrame:
    return pd.read_parquet(CACHE) if CACHE.exists() else build()


if __name__ == "__main__":
    tr = build()
    print(len(tr), "departures"); print(tr.groupby("ap")[TRAFFIC].mean().round(1).to_string())
