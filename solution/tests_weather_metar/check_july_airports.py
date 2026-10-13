#!/usr/bin/env python3
"""Detailed airport-by-airport breakdown of July 2025 weather residuals on uncovered flights.
Check whether Thunderstorm and Rain under-prediction is consistent across airports or concentrated at one hub.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "solution"
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
}, index=X.index.astype("int64"))

df = df[np.isfinite(df.y)].copy()

days = sorted(f.stem for f in (EXP / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in ("2025-01", "2025-07"))
e = pd.concat([adsb_features.load_day(d) for d in days]).sort_values("liftoff_dt")
e = e[~e.index.duplicated()]
e.index = e.index.astype("int64")

A7 = ["EDDF", "EHAM", "LEBL", "EDDM", "LSZH", "LEMD", "EGLL"]
df = df.join(e[["e_first"]], how="left")
df["covered"] = df.e_first.notna() & df.ap.isin(A7)

def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")

wx_frames = []
for st in sorted(df.ap.unique()):
    fp = WX / f"{st}.csv"
    d = pd.read_csv(fp, dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    wx = d["wxcodes"].fillna("").str.upper()
    sub_df = pd.DataFrame({
        "valid": d["valid"],
        "station": st,
        "tmpf": num(d["tmpf"]),
        "dwpf": num(d["dwpf"]),
        "vsby": num(d["vsby"]),
        "is_ts": wx.str.contains("TS", regex=True),
        "is_rain": wx.str.contains("RA|DZ", regex=True),
    }).dropna(subset=["valid", "tmpf"]).sort_values("valid")
    wx_frames.append(sub_df)

wx_all = pd.concat(wx_frames, ignore_index=True).sort_values("valid")

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

df["is_ts"] = merged["is_ts"].fillna(False).to_numpy().astype(bool)
df["is_rain"] = merged["is_rain"].fillna(False).to_numpy().astype(bool)

jul_unc = df[(df.month == 7) & (~df.covered)].copy()

print("\n=== JULY 2025 THUNDERSTORM BREAKDOWN BY AIRPORT (UNCOVERED) ===")
for ap in sorted(jul_unc.ap.unique()):
    ap_all = jul_unc[jul_unc.ap == ap]
    ap_ts = ap_all[ap_all.is_ts]
    if len(ap_ts) > 0:
        in_band_all = ap_all[ap_all.resid.abs() < 3600]
        in_band_ts = ap_ts[ap_ts.resid.abs() < 3600]
        delta = in_band_ts.resid.mean() - in_band_all.resid.mean()
        print(f"  {ap:6s}: n_ts={len(ap_ts):3d} / {len(ap_all):5d} | TS mean: {in_band_ts.resid.mean():+6.1f} s | normal: {in_band_all.resid.mean():+6.1f} s | delta: {delta:+6.1f} s | TS median: {in_band_ts.resid.median():+6.1f} s")

print("\n=== JULY 2025 RAIN (NON-TS) BREAKDOWN BY AIRPORT (UNCOVERED) ===")
for ap in sorted(jul_unc.ap.unique()):
    ap_all = jul_unc[jul_unc.ap == ap]
    ap_rain = ap_all[ap_all.is_rain & (~ap_all.is_ts)]
    if len(ap_rain) >= 10:
        in_band_all = ap_all[ap_all.resid.abs() < 3600]
        in_band_rain = ap_rain[ap_rain.resid.abs() < 3600]
        delta = in_band_rain.resid.mean() - in_band_all.resid.mean()
        print(f"  {ap:6s}: n_rain={len(ap_rain):4d} / {len(ap_all):5d} | rain mean: {in_band_rain.resid.mean():+6.1f} s | normal: {in_band_all.resid.mean():+6.1f} s | delta: {delta:+6.1f} s | rain median: {in_band_rain.resid.median():+6.1f} s")
