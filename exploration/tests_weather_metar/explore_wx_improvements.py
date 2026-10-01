#!/usr/bin/env python3
"""Explore weather and uncovered flight residuals on Jan and Jul 2025 (seasonal_full test).
Examine:
1. Frost vs Snow vs Freezing Rain vs Fog vs Wind in Jan 2025
2. Thunderstorms, heavy rain, extreme heat in Jul 2025
3. Airport-specific responses, especially for uncovered airports LFPG and LTFM.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "exploration"
WX = ROOT / "prc-2026-datasets-wx"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / "tests_adsb_full"))
import config
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE
import features as adsb_features

print("Loading seasonal_full test data...")
X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)

# Load base model predictions on seasonal_full test
tree_sf = np.load(CACHE / "tree_seasonal_full_test.npy")
mlp_sf = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt_sf = np.mean([np.load(EXP / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
base = (tree_sf + mlp_sf + ftt_sf) / 3.0

df = pd.DataFrame({
    "mvt_id": X.index.astype("int64"),
    "y": np.asarray(y, float),
    "base": base,
    "resid": np.asarray(y, float) - base,
    "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
    "valid": t.to_numpy(),
    "month": t.dt.month.to_numpy(),
    "day": t.dt.strftime("%Y-%m-%d").to_numpy(),
    "hour": t.dt.hour.to_numpy(),
}, index=X.index.astype("int64"))

# Filter out non-finite targets
df = df[np.isfinite(df.y)].copy()

# Identify ADS-B covered flights in Jan and Jul 2025
print("Loading ADS-B day parquets for Jan and Jul 2025...")
days = sorted(f.stem for f in (EXP / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in ("2025-01", "2025-07"))
e = pd.concat([adsb_features.load_day(d) for d in days]).sort_values("liftoff_dt")
e = e[~e.index.duplicated()]
e.index = e.index.astype("int64")

A7 = ["EDDF", "EHAM", "LEBL", "EDDM", "LSZH", "LEMD", "EGLL"]
df = df.join(e[["e_first"]], how="left")
df["covered"] = df.e_first.notna() & df.ap.isin(A7)

print(f"Total rows in seasonal_full test: {len(df)}")
print(f"January rows: {(df.month == 1).sum()} (covered: {((df.month == 1) & df.covered).sum()}, uncovered: {((df.month == 1) & ~df.covered).sum()})")
print(f"July rows: {(df.month == 7).sum()} (covered: {((df.month == 7) & df.covered).sum()}, uncovered: {((df.month == 7) & ~df.covered).sum()})")

# Load and parse METAR
print("Parsing METAR for all airports...")
def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")

wx_frames = []
for st in sorted(df.ap.unique()):
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
left = df[["mvt_id", "valid", "ap"]].rename(columns={"ap": "station"}).sort_values("valid")
merged = pd.merge_asof(
    left,
    wx_all,
    on="valid",
    by="station",
    direction="backward",
    tolerance=pd.Timedelta("90min"),
)
merged.index = merged["mvt_id"].astype("int64")
merged = merged.reindex(df.index)

for col in ["tmpf", "dwpf", "sknt", "vsby", "gust", "is_snow", "is_fz", "is_rain", "is_ts", "is_fog", "ceiling"]:
    df[col] = merged[col]

df["spread"] = df["tmpf"] - df["dwpf"]
df["is_frost"] = ((df["tmpf"] <= 34.0) & (df["spread"] <= 3.0)).fillna(False)

print("\n=== JANUARY 2025 UNCOVERED FLIGHTS RESIDUAL ANALYSIS ===")
jan_unc = df[(df.month == 1) & (~df.covered)].copy()

print(f"Total Jan uncovered flights: {len(jan_unc)}")
print(f"Overall Jan uncovered resid mean: {jan_unc.resid.mean():.1f} s, median: {jan_unc.resid.median():.1f} s")

# Weather condition buckets on Jan uncovered flights
conds = {
    "All Frost (v49 rule: tmpf<=34 & spread<=3)": jan_unc.is_frost.fillna(False),
    "Snow (SN/SG/IC/PL)": jan_unc.is_snow.fillna(False),
    "Freezing condition (FZ)": jan_unc.is_fz.fillna(False),
    "Snow OR Freezing": (jan_unc.is_snow | jan_unc.is_fz).fillna(False),
    "Heavy Fog (vsby < 1 mi or FG)": ((jan_unc.vsby < 1.0) | jan_unc.is_fog).fillna(False),
    "High Wind (gust >= 25 kt)": (jan_unc.gust >= 25.0).fillna(False),
    "Cold & Dry (tmpf <= 32 & spread > 5)": ((jan_unc.tmpf <= 32.0) & (jan_unc.spread > 5.0)).fillna(False),
    "Mild (tmpf > 40)": (jan_unc.tmpf > 40.0).fillna(False),
}

for name, mask in conds.items():
    sub = jan_unc[mask]
    if len(sub) > 0:
        # In-band (|resid| < 3600) to check bulk behavior vs extreme artifacts
        in_band = sub[sub.resid.abs() < 3600]
        print(f"\nCondition: {name}")
        print(f"  Count: {len(sub)} ({len(sub)/len(jan_unc):.1%})")
        print(f"  All resid mean: {sub.resid.mean():+.1f} s, median: {sub.resid.median():+.1f} s, RMSE: {np.sqrt((sub.resid**2).mean()):.1f}")
        print(f"  In-band (<3600s, n={len(in_band)}): mean: {in_band.resid.mean():+.1f} s, median: {in_band.resid.median():+.1f} s")

print("\n=== JANUARY 2025 FROST BY AIRPORT ===")
for ap in sorted(jan_unc.ap.unique()):
    ap_all = jan_unc[jan_unc.ap == ap]
    ap_frost = ap_all[ap_all.is_frost]
    if len(ap_frost) >= 20:
        in_band_all = ap_all[ap_all.resid.abs() < 3600]
        in_band_frost = ap_frost[ap_frost.resid.abs() < 3600]
        diff_mean = in_band_frost.resid.mean() - in_band_all.resid.mean()
        print(f"  {ap:6s}: n_frost={len(ap_frost):4d} / {len(ap_all):5d} | frost mean: {in_band_frost.resid.mean():+5.1f} s | normal: {in_band_all.resid.mean():+5.1f} s | delta: {diff_mean:+5.1f} s | frost median: {in_band_frost.resid.median():+5.1f} s")

print("\n=== JULY 2025 UNCOVERED FLIGHTS RESIDUAL ANALYSIS ===")
jul_unc = df[(df.month == 7) & (~df.covered)].copy()
print(f"Total Jul uncovered flights: {len(jul_unc)}")
print(f"Overall Jul uncovered resid mean: {jul_unc.resid.mean():.1f} s, median: {jul_unc.resid.median():.1f} s")

jul_conds = {
    "Thunderstorm (TS)": jul_unc.is_ts.fillna(False),
    "Rain (RA/DZ)": jul_unc.is_rain.fillna(False),
    "High Wind (gust >= 25 kt)": (jul_unc.gust >= 25.0).fillna(False),
    "Extreme Heat (tmpf >= 86F / 30C)": (jul_unc.tmpf >= 86.0).fillna(False),
    "Moderate Heat (tmpf 77-85F / 25-29C)": ((jul_unc.tmpf >= 77.0) & (jul_unc.tmpf < 86.0)).fillna(False),
    "Normal Summer (tmpf 60-76F)": ((jul_unc.tmpf >= 60.0) & (jul_unc.tmpf < 77.0)).fillna(False),
}

for name, mask in jul_conds.items():
    sub = jul_unc[mask]
    if len(sub) > 0:
        in_band = sub[sub.resid.abs() < 3600]
        print(f"\nCondition: {name}")
        print(f"  Count: {len(sub)} ({len(sub)/len(jul_unc):.1%})")
        print(f"  All resid mean: {sub.resid.mean():+.1f} s, median: {sub.resid.median():+.1f} s, RMSE: {np.sqrt((sub.resid**2).mean()):.1f}")
        print(f"  In-band (<3600s, n={len(in_band)}): mean: {in_band.resid.mean():+.1f} s, median: {in_band.resid.median():+.1f} s")
