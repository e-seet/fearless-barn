#!/usr/bin/env python3
"""Check the 428 July 2026 thunderstorm flights in v49:
- What does v49 currently predict for them?
- What are their movement times and flight details?
- If we add +180s (or +240s), what is the resulting taxi-time distribution?
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "exploration"
WX = ROOT / "prc-2026-datasets-wx"
SUB = EXP / "submissions"
sys.path.insert(0, str(EXP))
import config
from common import dataset as D

v49 = pd.read_parquet(SUB / "fearless_barn_v49.parquet")
v42 = pd.read_parquet(SUB / "fearless_barn_v42.parquet")
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

_, meta = D.load_ranking_features("with_aobt")
mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month

uncovered = (v49[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
is_jul = (month.to_numpy() == 7)

# Load METAR
wx_frames = []
for st in sorted(adep.unique()):
    fp = WX / f"{st}.csv"
    d = pd.read_csv(fp, dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    wx = d["wxcodes"].fillna("").str.upper()
    sub_df = pd.DataFrame({
        "valid": d["valid"],
        "station": st,
        "is_ts": wx.str.contains("TS", regex=True),
    }).dropna(subset=["valid"]).sort_values("valid")
    wx_frames.append(sub_df)

wx_all = pd.concat(wx_frames, ignore_index=True).sort_values("valid")

left = pd.DataFrame({
    "mvt_id": tmpl[config.ID].to_numpy(),
    "valid": mvt_time.to_numpy(),
    "station": adep.to_numpy(),
}).sort_values("valid")

merged = pd.merge_asof(
    left,
    wx_all,
    on="valid",
    by="station",
    direction="backward",
    tolerance=pd.Timedelta("90min"),
).set_index("mvt_id").reindex(tmpl[config.ID].to_numpy())

is_ts = merged["is_ts"].fillna(False).to_numpy(dtype=bool)
ts_mask = is_jul & uncovered & is_ts

print(f"Total July Thunderstorm uncovered flights: {ts_mask.sum()}")
print("\nAirport breakdown:")
for ap, cnt in adep[ts_mask].value_counts().items():
    print(f"  {ap:6s}: {cnt:4d}")

ts_preds = v49[config.TARGET][ts_mask]
print(f"\nCurrent v49 predictions on these {len(ts_preds)} flights:")
print(f"  mean:   {ts_preds.mean():.1f} s ({ts_preds.mean()/60:.1f} min)")
print(f"  median: {ts_preds.median():.1f} s ({ts_preds.median()/60:.1f} min)")
print(f"  min:    {ts_preds.min():.1f} s")
print(f"  p25:    {ts_preds.quantile(0.25):.1f} s")
print(f"  p75:    {ts_preds.quantile(0.75):.1f} s")
print(f"  max:    {ts_preds.max():.1f} s")

# Look at LTFM specifically
ltfm_ts = ts_mask & (adep == "LTFM")
print(f"\nLTFM July TS predictions (n={ltfm_ts.sum()}):")
p_ltfm = v49[config.TARGET][ltfm_ts]
print(f"  mean:   {p_ltfm.mean():.1f} s ({p_ltfm.mean()/60:.1f} min)")
print(f"  median: {p_ltfm.median():.1f} s ({p_ltfm.median()/60:.1f} min)")
print(f"  p25:    {p_ltfm.quantile(0.25):.1f} s, p75: {p_ltfm.quantile(0.75):.1f} s, max: {p_ltfm.max():.1f} s")
