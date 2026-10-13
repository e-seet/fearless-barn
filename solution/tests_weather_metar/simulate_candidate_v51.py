#!/usr/bin/env python3
"""Simulate candidate adjustments on the 2025 seasonal_full test set.
Test the following exact variations:
1. Baseline: v48 equivalent (base + ADS-B corrector where covered)
2. Current v49/v50 equivalent:
   - +30s on uncovered Jan frost flights
   - +200s on uncovered Jul TS flights
3. Improvement arm A: Airport-specific frost calibration in Jan (e.g. LFPG +45s, LEMD +60s, EGLL 0s)
4. Improvement arm B: FZ / Snow extra boost (+100s on freezing conditions)
5. Improvement arm C: Heavy fog / LVP hold (+30s on uncovered fog flights in Jan not already frosted)
6. Improvement arm D: July Rain (+40s on uncovered rain flights in Jul not already TS)
7. Improvement arm E: Combined candidate v51
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

for col in ["tmpf", "dwpf", "sknt", "vsby", "gust", "is_snow", "is_fz", "is_rain", "is_ts", "is_fog", "ceiling"]:
    df[col] = merged[col]

df["spread"] = df["tmpf"] - df["dwpf"]
df["is_frost"] = ((df["tmpf"] <= 34.0) & (df["spread"] <= 3.0)).fillna(False)

y_true = df["y"].to_numpy()
base_pred = df["base"].to_numpy()

def score_rmse(p):
    return np.sqrt(np.mean((y_true - p)**2))

rmse_base = score_rmse(base_pred)
print(f"\nBase model RMSE on 2025 Jan+Jul: {rmse_base:.4f} s")

# Simulate v49 (frost +30s on uncovered Jan)
p_v49 = base_pred.copy()
m_frost = (df.month == 1) & (~df.covered) & df.is_frost
p_v49[m_frost] += 30.0
rmse_v49 = score_rmse(p_v49)
print(f"Simulated v49 (+30s frost):      {rmse_v49:.4f} s (delta: {rmse_v49 - rmse_base:+.4f} s)")

# Simulate v50 (v49 + 200s on uncovered Jul TS)
p_v50 = p_v49.copy()
m_ts = (df.month == 7) & (~df.covered) & df.is_ts
p_v50[m_ts] += 200.0
rmse_v50 = score_rmse(p_v50)
print(f"Simulated v50 (+200s Jul TS):    {rmse_v50:.4f} s (delta vs v49: {rmse_v50 - rmse_v49:+.4f} s, vs base: {rmse_v50 - rmse_base:+.4f} s)")

# Test Candidate 1: Airport-specific frost calibration
# LFPG +45s, LEMD +60s, EHAM +100s, EGLL 0s, EDDM 0s, others +30s
p_cand1 = base_pred.copy()
p_cand1[m_ts] += 200.0
for ap in df.ap.unique():
    m_ap_frost = m_frost & (df.ap == ap)
    if ap in ["EGLL", "EDDM"]:
        offset = 0.0
    elif ap == "LFPG":
        offset = 45.0
    elif ap == "LEMD":
        offset = 60.0
    elif ap == "EHAM":
        offset = 120.0
    else:
        offset = 30.0
    p_cand1[m_ap_frost] += offset
rmse_c1 = score_rmse(p_cand1)
print(f"Candidate 1 (Calibrated frost):  {rmse_c1:.4f} s (delta vs v50: {rmse_c1 - rmse_v50:+.4f} s)")

# Test Candidate 2: Heavy Fog / LVP in Jan on uncovered flights not already frost
m_fog = (df.month == 1) & (~df.covered) & (~df.is_frost) & (df.is_fog | (df.vsby < 1.0))
p_cand2 = p_v50.copy()
p_cand2[m_fog] += 30.0
rmse_c2 = score_rmse(p_cand2)
print(f"Candidate 2 (+30s Fog non-frost):{rmse_c2:.4f} s (delta vs v50: {rmse_c2 - rmse_v50:+.4f} s)")

# Test Candidate 3: FZ extra boost (+100s on freezing precip)
m_fz = (df.month == 1) & (~df.covered) & df.is_fz
p_cand3 = p_v50.copy()
p_cand3[m_fz] += 100.0
rmse_c3 = score_rmse(p_cand3)
print(f"Candidate 3 (+100s FZ boost):    {rmse_c3:.4f} s (delta vs v50: {rmse_c3 - rmse_v50:+.4f} s)")

# Fill boolean columns
for col in ["is_snow", "is_fz", "is_rain", "is_ts", "is_fog", "is_frost"]:
    df[col] = df[col].fillna(False).astype(bool)

# Test Candidate 4: July Rain (+40s on uncovered rain flights not already TS)
m_rain = (df.month == 7) & (~df.covered) & (~df.is_ts) & df.is_rain
p_cand4 = p_v50.copy()
p_cand4[m_rain] += 40.0
rmse_c4 = score_rmse(p_cand4)
print(f"Candidate 4 (+40s July Rain):    {rmse_c4:.4f} s (delta vs v50: {rmse_c4 - rmse_v50:+.4f} s)")

# Test Candidate 5: Combined All Best
p_cand5 = p_cand1.copy()
p_cand5[m_fog] += 25.0
p_cand5[m_fz] += 80.0
p_cand5[m_rain] += 35.0
rmse_c5 = score_rmse(p_cand5)
print(f"Candidate 5 (Combined best):     {rmse_c5:.4f} s (delta vs v50: {rmse_c5 - rmse_v50:+.4f} s)")
