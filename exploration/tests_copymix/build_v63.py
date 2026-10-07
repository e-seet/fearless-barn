#!/usr/bin/env python3
"""v63 = lobt_clip((1-w_a)*S2 + w_a*M + (v61 - S2)): the copy blend is applied to the stage-2 output (what was validated),
the stage 3-5 deltas of v61 (weather offsets, LIRF and LFPG correctors) are added back unchanged, then the LOBT window."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import config
from copymix import lobt_clip
v61 = pd.read_parquet(ROOT / "submissions" / "fearless_barn_v61.parquet")
s2 = pd.read_parquet(HERE / "v61_s2.parquet")
assert (v61[config.ID].to_numpy() == s2[config.ID].to_numpy()).all()
Mr = pd.read_parquet(HERE / "cache" / "M_ranking.parquet")
ids = v61[config.ID].astype("int64").to_numpy(); Mr.index = Mr.index.astype("int64"); Mr = Mr.reindex(ids)
W = json.loads((HERE / "weights_v63.json").read_text())["weights_per_airport"]
w = Mr.ap.map(W).fillna(0.0).to_numpy().copy()
S2 = s2[config.TARGET].to_numpy(float); F = v61[config.TARGET].to_numpy(float); delta = F - S2
M = Mr.M.to_numpy(); bad = ~np.isfinite(M); w[bad] = 0.0; M = np.where(bad, S2, M)
pred = (1 - w) * S2 + w * M + delta
pred = lobt_clip(pred, Mr.mvt.to_numpy(), Mr.lobt.to_numpy())
pred = np.clip(np.rint(pred), 60, 86400)
out = v61[[config.ID]].copy(); out[config.TARGET] = pred.astype("int32")
dst = ROOT / "submissions" / "fearless_barn_v63.parquet"; out.to_parquet(dst, index=False)
v62 = pd.read_parquet(ROOT / "submissions" / "fearless_barn_v62.parquet")[config.TARGET].to_numpy(float)
dd = pred - F
print(f"wrote {dst}: {len(out)} rows | M NaN fallback {int(bad.sum())} | rows changed vs v61 {int((dd != 0).sum())} | "
      f"RMS change vs v61 {np.sqrt(np.mean(dd ** 2)):.2f} s | vs v62 {np.sqrt(np.mean((pred - v62) ** 2)):.2f} s | mean change {dd.mean():+.2f} s")
print(pd.DataFrame({"ap": Mr.ap.to_numpy(), "mon": Mr.mon.to_numpy(), "d": dd}).groupby(["ap", "mon"]).d.agg(["mean", lambda x: float(np.sqrt(np.mean(x ** 2)))]).round(2).to_string())
