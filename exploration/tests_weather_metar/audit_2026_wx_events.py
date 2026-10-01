#!/usr/bin/env python3
"""Audit 2026 scored set for weather events on uncovered flights:
1. July 2026: Thunderstorms (TS) and Summer Rain (RA/DZ)
2. January 2026: Freezing precipitation (FZ), Heavy Fog (FG / vsby < 1 mi)
Check flight counts, airport breakdown, and overlap with v49.
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

# 1. Load ranking metadata and submissions
v48 = pd.read_parquet(SUB / "fearless_barn_v48.parquet")
v42 = pd.read_parquet(SUB / "fearless_barn_v42.parquet")
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

_, meta = D.load_ranking_features("with_aobt")
assert (meta.index == tmpl[config.ID]).all()

mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month

uncovered = (v48[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
is_jan = (month.to_numpy() == 1)
is_jul = (month.to_numpy() == 7)

# 2. Parse METAR for all airports
def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")

wx_frames = []
for st in sorted(adep.unique()):
    fp = WX / f"{st}.csv"
    d = pd.read_csv(fp, dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    wx = d["wxcodes"].fillna("").str.upper()
    skyl = num(d["skyl1"])
    skyc = d["skyc1"].fillna("").str.upper()
    low_cloud = skyc.isin(["BKN", "OVC", "VV"])
    ceiling = np.where(low_cloud, skyl, 25000.0)
    
    sub_df = pd.DataFrame({
        "valid": d["valid"],
        "station": st,
        "tmpf": num(d["tmpf"]),
        "dwpf": num(d["dwpf"]),
        "sknt": num(d["sknt"]),
        "vsby": num(d["vsby"]),
        "gust": num(d["gust"]).fillna(num(d["sknt"])),
        "is_snow": wx.str.contains("SN|SG|IC|PL", regex=True),
        "is_fz": wx.str.contains("FZ", regex=True),
        "is_rain": wx.str.contains("RA|DZ", regex=True),
        "is_ts": wx.str.contains("TS", regex=True),
        "is_fog": wx.str.contains(r"\bFG\b|\bBR\b|FZFG|MIFG", regex=True),
        "ceiling": ceiling,
    }).dropna(subset=["valid", "tmpf"]).sort_values("valid")
    wx_frames.append(sub_df)

wx_all = pd.concat(wx_frames, ignore_index=True).sort_values("valid")

# As-of join
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

# Boolean masks
spread = merged["tmpf"] - merged["dwpf"]
is_frost = ((merged["tmpf"] <= 34.0) & (spread <= 3.0)).fillna(False).to_numpy()
is_snow = merged["is_snow"].fillna(False).to_numpy()
is_fz = merged["is_fz"].fillna(False).to_numpy()
is_ts = merged["is_ts"].fillna(False).to_numpy()
is_rain = merged["is_rain"].fillna(False).to_numpy()
is_heavy_fog = ((merged["vsby"] < 1.0) | (merged["is_fog"] & (merged["vsby"] <= 2.0))).fillna(False).to_numpy()

print("=== 2026 SCORED SET WEATHER EVENT COUNTS ON UNCOVERED FLIGHTS ===")
print(f"Total scored rows: {len(tmpl)}")
print(f"Uncovered flights: {uncovered.sum()} (Jan: {(is_jan & uncovered).sum()}, Jul: {(is_jul & uncovered).sum()})")

print("\n--- JANUARY 2026 UNCOVERED ---")
jan_unc = is_jan & uncovered
print(f"Jan Uncovered total: {jan_unc.sum()}")
print(f"  Frost (v49 target): {(jan_unc & is_frost).sum()}")
print(f"  Freezing precip (FZ): {(jan_unc & is_fz).sum()}")
print(f"  Snow (SN/SG/PL): {(jan_unc & is_snow).sum()}")
print(f"  Heavy Fog (vsby < 1 mi or FG <= 2 mi): {(jan_unc & is_heavy_fog).sum()}")
print(f"  Heavy Fog NOT in Frost: {(jan_unc & is_heavy_fog & ~is_frost).sum()}")

print("\n--- JULY 2026 UNCOVERED ---")
jul_unc = is_jul & uncovered
print(f"Jul Uncovered total: {jul_unc.sum()}")
print(f"  Thunderstorm (TS): {(jul_unc & is_ts).sum()}")
print(f"  Rain (RA/DZ): {(jul_unc & is_rain).sum()}")
print(f"  Rain NOT TS: {(jul_unc & is_rain & ~is_ts).sum()}")

print("\nBreakdown of July Thunderstorms on uncovered flights by airport:")
for ap, cnt in adep[jul_unc & is_ts].value_counts().items():
    print(f"  {ap:6s}: {cnt:4d}")

print("\nBreakdown of July Rain on uncovered flights by airport:")
for ap, cnt in adep[jul_unc & is_rain].value_counts().items():
    print(f"  {ap:6s}: {cnt:4d}")

print("\nBreakdown of Jan Heavy Fog (NOT frost) on uncovered flights by airport:")
for ap, cnt in adep[jan_unc & is_heavy_fog & ~is_frost].value_counts().items():
    print(f"  {ap:6s}: {cnt:4d}")
