#!/usr/bin/env python3
"""Inspect the 2026 test set (ranking.parquet) for high-potential weather signals:
1. Freezing condition (FZ) in January 2026 (uncovered)
2. Heavy Fog (FG or vsby < 1.0) in January 2026 (uncovered)
3. July Rain (RA / DZ) in July 2026 (uncovered)
4. Airport-specific frost calibration for v49 (e.g. LFPG, LEMD vs EGLL)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "solution"; exec(open(ROOT / "tools" / "repo_layout.py").read()); EXP = dual(EXP)
WX = ROOT / "prc-2026-datasets-wx"
SUB = EXP / "submissions"
sys.path.insert(0, str(EXP))
import config
from common import dataset as D

v50 = pd.read_parquet(SUB / "fearless_barn_v50.parquet")
v42 = pd.read_parquet(SUB / "fearless_barn_v42.parquet")
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

_, meta = D.load_ranking_features("with_aobt")
mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month

uncovered = (v50[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
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

merged["spread"] = merged["tmpf"] - merged["dwpf"]
merged["is_frost"] = ((merged["tmpf"] <= 34.0) & (merged["spread"] <= 3.0)).fillna(False)

print(f"Total 2026 test flights: {len(tmpl)}")
print(f"Uncovered flights: {uncovered.sum()} ({uncovered.sum()/len(tmpl):.1%})")
print(f"Uncovered Jan: {(is_jan & uncovered).sum()} | Uncovered Jul: {(is_jul & uncovered).sum()}")

print("\n--- 1. JANUARY 2026 UNCOVERED SIGNALS ---")
jan_unc = is_jan & uncovered
n_frost = (jan_unc & merged.is_frost).sum()
n_fz = (jan_unc & merged.is_fz).sum()
n_snow = (jan_unc & merged.is_snow).sum()
n_fog = (jan_unc & (merged.is_fog | (merged.vsby < 1.0))).sum()
n_dense_fog = (jan_unc & (merged.vsby < 0.5)).sum()

print(f"Jan Frost (already modified in v49/v50 by +30s): {n_frost}")
print(f"Jan Freezing condition (FZ): {n_fz}")
if n_fz > 0:
    for ap, cnt in adep[jan_unc & merged.is_fz].value_counts().items():
        print(f"    {ap}: {cnt}")
print(f"Jan Snow: {n_snow}")
print(f"Jan Fog/BR: {n_fog}")
print(f"Jan Dense Fog (vsby < 0.5 mi): {n_dense_fog}")
if n_dense_fog > 0:
    for ap, cnt in adep[jan_unc & (merged.vsby < 0.5)].value_counts().items():
        print(f"    {ap}: {cnt}")

print("\n--- 2. JULY 2026 UNCOVERED SIGNALS ---")
jul_unc = is_jul & uncovered
n_ts = (jul_unc & merged.is_ts).sum()
n_rain = (jul_unc & merged.is_rain).sum()

print(f"Jul Thunderstorms (already modified in v50 by +200s): {n_ts}")
print(f"Jul Rain (RA / DZ): {n_rain}")
if n_rain > 0:
    for ap, cnt in adep[jul_unc & merged.is_rain].value_counts().items():
        print(f"    {ap}: {cnt}")
