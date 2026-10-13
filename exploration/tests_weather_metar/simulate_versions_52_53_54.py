#!/usr/bin/env python3
"""Simulate v52, v53, v54 candidates on the 2025 seasonal_full test set:
- v52: Conservative Frost & De-icing Specialist (Calibrated frost + FZ, no fog/rain)
- v53: Aggressive Convective & Extended Severe Weather (TS +320s at LTFM, FZ +120s, SN +50s, calibrated frost)
- v54: LVP & Peak-Hour Congestion Weather Interaction (Fog/Snow with hour-of-day weighting)
Compare all against v50 and v51.
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

# Load base model predictions on seasonal_full test
tree_sf = np.load(CACHE / "tree_seasonal_full_test.npy")
mlp_sf = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt_sf = np.mean([np.load(EXP / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
base = (tree_sf + mlp_sf + ftt_sf) / 3.0

df = pd.DataFrame({
    "mvt_id": X.index.astype("int64"),
    "y": np.asarray(y, float),
    "base": base,
    "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
    "valid": t.to_numpy(),
    "month": t.dt.month.to_numpy(),
    "hour": t.dt.hour.to_numpy(),
}, index=X.index.astype("int64"))

df = df[np.isfinite(df.y)].copy()

# Load ADS-B day parquets for Jan and Jul 2025
days = sorted(f.stem for f in (EXP / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in ("2025-01", "2025-07"))
e = pd.concat([adsb_features.load_day(d) for d in days]).sort_values("liftoff_dt")
e = e[~e.index.duplicated()]
e.index = e.index.astype("int64")

A7 = ["EDDF", "EHAM", "LEBL", "EDDM", "LSZH", "LEMD", "EGLL"]
df = df.join(e[["e_first"]], how="left")
df["covered"] = df.e_first.notna() & df.ap.isin(A7)

# Parse METAR
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
        "is_rain": wx.str.contains(r"\bRA\b|\bDZ\b|\+RA|-RA|\+DZ|-DZ", regex=True),
        "is_ts": wx.str.contains("TS", regex=True),
        "is_fog": wx.str.contains(r"\bFG\b|\bBR\b|FZFG|MIFG", regex=True),
        "ceiling": ceiling,
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

for col in ["tmpf", "dwpf", "sknt", "vsby", "gust", "is_snow", "is_fz", "is_rain", "is_ts", "is_fog"]:
    if col in ["tmpf", "dwpf", "sknt", "vsby", "gust"]:
        df[col] = num(merged[col])
    else:
        df[col] = merged[col].fillna(False).astype(bool)

df["spread"] = df["tmpf"] - df["dwpf"]
df["is_frost"] = ((df["tmpf"] <= 34.0) & (df["spread"] <= 3.0)).fillna(False).astype(bool)

y_true = df["y"].to_numpy()
base_pred = df["base"].to_numpy()

def score_rmse(p):
    return np.sqrt(np.mean((y_true - p)**2))

# v50 reproduction
p_v50 = base_pred.copy()
m_frost = (df.month == 1) & (~df.covered) & df.is_frost
p_v50[m_frost] += 30.0
m_ts = (df.month == 7) & (~df.covered) & df.is_ts
p_v50[m_ts] += 200.0
rmse_v50 = score_rmse(p_v50)
print(f"v50 baseline RMSE: {rmse_v50:.4f} s")

# v51 reproduction
p_v51 = base_pred.copy()
p_v51[m_ts] += 200.0
for ap in df.ap.unique():
    m_ap = m_frost & (df.ap == ap)
    if ap == "LFPG": p_v51[m_ap] += 45.0
    elif ap == "LEMD": p_v51[m_ap] += 60.0
    elif ap == "EHAM": p_v51[m_ap] += 80.0
    elif ap in ["EGLL", "EDDM"]: pass
    else: p_v51[m_ap] += 30.0

m_fz = (df.month == 1) & (~df.covered) & df.is_fz
p_v51[m_fz] += 80.0
m_fog = (df.month == 1) & (~df.covered) & (~df.is_frost) & (df.is_fog | (df.vsby < 1.0))
p_v51[m_fog] += 25.0
m_rain = (df.month == 7) & (~df.covered) & (~df.is_ts) & df.is_rain
p_v51[m_rain] += 35.0
rmse_v51 = score_rmse(p_v51)
print(f"v51 RMSE:         {rmse_v51:.4f} s (delta vs v50: {rmse_v51 - rmse_v50:+.4f} s)")

# v52: Conservative De-icing Specialist (Calibrated Frost + FZ only)
p_v52 = base_pred.copy()
p_v52[m_ts] += 200.0
for ap in df.ap.unique():
    m_ap = m_frost & (df.ap == ap)
    if ap == "LFPG": p_v52[m_ap] += 45.0
    elif ap == "LEMD": p_v52[m_ap] += 60.0
    elif ap == "EHAM": p_v52[m_ap] += 80.0
    elif ap in ["EGLL", "EDDM"]: pass
    else: p_v52[m_ap] += 30.0
p_v52[m_fz] += 80.0
rmse_v52 = score_rmse(p_v52)
print(f"v52 RMSE (De-icing): {rmse_v52:.4f} s (delta vs v50: {rmse_v52 - rmse_v50:+.4f} s)")

# v53: Aggressive Convective & Severe Storm
p_v53 = p_v51.copy()
# Increase LTFM TS from 200s to 320s
m_ltfm_ts = m_ts & (df.ap == "LTFM")
p_v53[m_ltfm_ts] += 120.0  # total +320s
# Freezing precip boost from 80s to 120s
p_v53[m_fz] += 40.0        # total +120s
# Snow hold
m_snow = (df.month == 1) & (~df.covered) & (~df.is_frost) & df.is_snow
p_v53[m_snow] += 40.0
rmse_v53 = score_rmse(p_v53)
print(f"v53 RMSE (Aggressive): {rmse_v53:.4f} s (delta vs v50: {rmse_v53 - rmse_v50:+.4f} s)")

# v54: Peak-Hour Low-Visibility Congestion Interaction
p_v54 = p_v51.copy()
is_peak = (df.hour >= 6) & (df.hour <= 10)
m_peak_fog = m_fog & is_peak
p_v54[m_peak_fog] += 25.0  # total +50s during morning rush hour fog
rmse_v54 = score_rmse(p_v54)
print(f"v54 RMSE (Peak LVP):   {rmse_v54:.4f} s (delta vs v50: {rmse_v54 - rmse_v50:+.4f} s)")
