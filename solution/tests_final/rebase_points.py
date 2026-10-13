#!/usr/bin/env python3
"""Point files for the final round, consistent with final_ranking.parquet's takeoff times.

The Jan/Jul 2026 day files were cut against ranking.parquet; final_ranking.parquet shifts takeoff times by up
to 12 s. Each point's t_rel (seconds from the movement's takeoff) is re-based: t_rel + (T_old - T_new).
2026-01-31 is taken from a re-fetch against final_ranking (it must also hold 1 Feb early takeoffs).
Feb/Jun files were cut against final_ranking and are copied unchanged. Output: tests_final/points_final/."""
import shutil
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
SRC, OUT = ROOT / "tests_adsb_full" / "points", HERE / "points_final"; OUT.mkdir(exist_ok=True)
cols = ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]
def T(f):
    d = pd.read_parquet(f, columns=cols); d = d[d.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt")
    return pd.Series(pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy() / 1e9,
                     index=d.MVT_ID_mvt.to_numpy())
old, new = T(ROOT.parent / "prc-2026-datasets" / "ranking.parquet"), T(ROOT.parent / "prc-2026-datasets" / "final_ranking.parquet")
shift = (old - new.reindex(old.index)).dropna()
r = pd.read_parquet(ROOT.parent / "prc-2026-datasets" / "final_ranking.parquet", columns=cols)
t = pd.to_datetime(r[r.PHASE_mvt.eq("DEP")].MVT_TIME_UTC_mvt, utc=True)
days = sorted(set(t.dt.strftime("%Y-%m-%d")) | set((t - pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%d")))
n_shift = 0
for d in days:
    src = HERE / "points_refetch" / f"{d}.parquet" if d in ("2026-01-31", "2025-12-31") else SRC / f"{d}.parquet"
    if d[:7] in ("2026-02", "2026-06") or d in ("2026-01-31", "2025-12-31", "2026-05-31"):
        shutil.copy2(src, OUT / f"{d}.parquet"); continue
    p = pd.read_parquet(src)
    s = shift.reindex(p.MVT_ID_mvt.to_numpy()).to_numpy()
    k = np.isfinite(s); p.loc[k, "t_rel"] = p.t_rel.to_numpy()[k] + s[k]; n_shift += int((s[k] != 0).sum())
    p.to_parquet(OUT / f"{d}.parquet", index=False)
print(f"{len(days)} day files in {OUT}; points re-based {n_shift}; movements with a non-zero shift {(shift != 0).sum()} / {len(shift)}")
