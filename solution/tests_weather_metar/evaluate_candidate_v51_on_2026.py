#!/usr/bin/env python3
"""Evaluate Candidate 5 on the actual 2026 competition test set (ranking.parquet):
Check:
- How many flights receive calibrated frost adjustment vs v50
- How many flights receive fog / LVP adjustment
- How many flights receive freezing precip (FZ) boost
- How many flights receive July rain adjustment
- Summary of total delta to v50
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "solution"
WX = ROOT / "prc-2026-datasets-wx"
SUB = EXP / "submissions"
sys.path.insert(0, str(EXP))
import config
from common import dataset as D

v50 = pd.read_parquet(SUB / "fearless_barn_v50.parquet")
v48 = pd.read_parquet(SUB / "fearless_barn_v48.parquet")
v42 = pd.read_parquet(SUB / "fearless_barn_v42.parquet")
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

_, meta = D.load_ranking_features("with_aobt")
mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month

# True uncovered flights across all 2026:
uncovered = (v48[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
is_jan = (month.to_numpy() == 1)
is_jul = (month.to_numpy() == 7)

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
        "is_rain": wx.str.contains(r"\bRA\b|\bDZ\b|\+RA|-RA|\+DZ|-DZ", regex=True),
        "is_ts": wx.str.contains("TS", regex=True),
        "is_fog": wx.str.contains(r"\bFG\b|\bBR\b|FZFG|MIFG", regex=True),
        "ceiling": ceiling,
    }).dropna(subset=["valid", "tmpf"]).sort_values("valid")
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

for col in ["tmpf", "dwpf", "sknt", "vsby", "gust", "is_snow", "is_fz", "is_rain", "is_ts", "is_fog"]:
    if col in ["tmpf", "dwpf", "sknt", "vsby", "gust"]:
        merged[col] = num(merged[col])
    else:
        merged[col] = merged[col].fillna(False).astype(bool)

merged["spread"] = merged["tmpf"] - merged["dwpf"]
merged["is_frost"] = ((merged["tmpf"] <= 34.0) & (merged["spread"] <= 3.0)).fillna(False).astype(bool)

# Build candidate v51 starting from v50
p_v50 = v50[config.TARGET].to_numpy().astype(float)
p_v51 = p_v50.copy()

# 1. Airport calibration on Jan frost (v49 previously applied flat +30s)
# Differences relative to +30s:
# LFPG: +15s (making +45s)
# LEMD: +30s (making +60s)
# EHAM: +50s (making +80s)
# EGLL: -30s (making 0s, undoing v49)
# EDDM: -30s (making 0s, undoing v49)
m_frost = is_jan & uncovered & merged["is_frost"]
n_frost_adj = 0
for ap in np.unique(adep):
    m_ap = m_frost & (adep == ap)
    cnt = m_ap.sum()
    if cnt == 0: continue
    if ap == "LFPG":
        p_v51[m_ap] += 15.0
        n_frost_adj += cnt
        print(f"  LFPG frost adjustment (+15s beyond v49): {cnt} flights")
    elif ap == "LEMD":
        p_v51[m_ap] += 30.0
        n_frost_adj += cnt
        print(f"  LEMD frost adjustment (+30s beyond v49): {cnt} flights")
    elif ap == "EHAM":
        p_v51[m_ap] += 50.0
        n_frost_adj += cnt
        print(f"  EHAM frost adjustment (+50s beyond v49): {cnt} flights")
    elif ap in ["EGLL", "EDDM"]:
        p_v51[m_ap] -= 30.0
        n_frost_adj += cnt
        print(f"  {ap} frost adjustment (-30s reverting v49): {cnt} flights")

# 2. Freezing precipitation boost (FZ) in Jan
m_fz = is_jan & uncovered & merged["is_fz"]
print(f"  Freezing precipitation (FZ) in Jan: {m_fz.sum()} flights")
if m_fz.sum() > 0:
    p_v51[m_fz] += 80.0

# 3. Dense Fog / LVP hold in Jan on non-frost flights
m_fog = is_jan & uncovered & (~merged["is_frost"]) & (merged["is_fog"] | (merged["vsby"] < 1.0))
print(f"  Non-frost fog flights in Jan (+25s): {m_fog.sum()} flights")
p_v51[m_fog] += 25.0

# 4. July rain on non-TS flights
m_rain = is_jul & uncovered & (~merged["is_ts"]) & merged["is_rain"]
print(f"  Non-TS rain flights in Jul (+35s): {m_rain.sum()} flights")
p_v51[m_rain] += 35.0

diff = p_v51 - p_v50
changed = (diff != 0)
print(f"\nTotal flights modified in Candidate v51 vs v50: {changed.sum()} ({changed.mean():.2%})")
print(f"Diff stats on modified flights:")
print(f"  mean diff:   {diff[changed].mean():+.1f} s")
print(f"  median diff: {np.median(diff[changed]):+.1f} s")
print(f"  min diff:    {diff[changed].min():+.1f} s")
print(f"  max diff:    {diff[changed].max():+.1f} s")
print(f"  RMSE diff to v50: {np.sqrt(np.mean(diff**2)):.4f} s")
