#!/usr/bin/env python3
"""Evaluate weather adjustments on held-out seasonal_full test set (Jan + Jul 2025).
Measure exact pooled RMSE change across the entire 344,419 rows of seasonal_full test:
1. Frost adjustment (+30s, like v49)
2. Higher frost adjustment for LFPG / EDDF / EHAM
3. July Thunderstorm adjustment on uncovered flights (+150s to +300s)
4. July Rain adjustment on uncovered flights (+40s)
5. Heavy Fog adjustment (+30s)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "solution"; exec(open(ROOT / "tools" / "repo_layout.py").read()); EXP = dual(EXP)
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

# Filter out non-finite targets
df = df[np.isfinite(df.y)].copy()

# Identify ADS-B covered flights
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
    sub_df = pd.DataFrame({
        "valid": d["valid"],
        "station": st,
        "tmpf": num(d["tmpf"]),
        "dwpf": num(d["dwpf"]),
        "vsby": num(d["vsby"]),
        "is_ts": wx.str.contains("TS", regex=True),
        "is_rain": wx.str.contains("RA|DZ", regex=True),
        "is_fog": wx.str.contains(r"\bFG\b|\bBR\b|FZFG|MIFG", regex=True),
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

spread = merged["tmpf"] - merged["dwpf"]
is_frost = ((merged["tmpf"] <= 34.0) & (spread <= 3.0)).fillna(False).to_numpy().astype(bool)
is_ts = merged["is_ts"].fillna(False).to_numpy().astype(bool)
is_rain = merged["is_rain"].fillna(False).to_numpy().astype(bool)
is_heavy_fog = ((merged["vsby"] < 1.0) | (merged["is_fog"] & (merged["vsby"] <= 2.0))).fillna(False).to_numpy().astype(bool)

unc = (~df["covered"].to_numpy()).astype(bool)
is_jan = (df["month"].to_numpy() == 1).astype(bool)
is_jul = (df["month"].to_numpy() == 7).astype(bool)
y_true = df["y"].to_numpy()
base_pred = df["base"].to_numpy()

def score_rmse(pred):
    return np.sqrt(np.mean((y_true - np.clip(np.rint(pred), 60, 86400))**2))

baseline_rmse = score_rmse(base_pred)
print(f"\nBaseline seasonal_full test RMSE (clean base): {baseline_rmse:.4f}")

# 1. Test Frost alone (+30s)
frost_mask = is_jan & unc & is_frost
p1 = base_pred.copy()
p1[frost_mask] += 30.0
r1 = score_rmse(p1)
print(f"1. Frost +30s (n={frost_mask.sum()}): RMSE {r1:.4f} (delta {r1 - baseline_rmse:+.4f} s)")

# 2. Test Frost tuning: LFPG +40s, EDDF +30s, EHAM +50s, LSZH +35s, LTFM +35s, exclude EGLL and EDDM
p2 = base_pred.copy()
for ap, delta in [("LFPG", 40.0), ("EDDF", 30.0), ("EHAM", 50.0), ("LSZH", 35.0), ("LTFM", 35.0), ("LEMD", 40.0)]:
    m = frost_mask & (df["ap"].to_numpy() == ap)
    p2[m] += delta
r2 = score_rmse(p2)
print(f"2. Calibrated Frost per airport (n={(frost_mask & df['ap'].isin(['LFPG', 'EDDF', 'EHAM', 'LSZH', 'LTFM', 'LEMD'])).sum()}): RMSE {r2:.4f} (delta {r2 - baseline_rmse:+.4f} s)")

# 3. Test July Thunderstorm on uncovered flights
ts_mask = is_jul & unc & is_ts
for ts_delta in [60.0, 120.0, 180.0, 240.0]:
    p3 = base_pred.copy()
    p3[ts_mask] += ts_delta
    r3 = score_rmse(p3)
    print(f"3. July TS +{ts_delta:.0f}s (n={ts_mask.sum()}): RMSE {r3:.4f} (delta {r3 - baseline_rmse:+.4f} s)")

# 4. Test July Rain on uncovered flights
rain_mask = is_jul & unc & is_rain & (~is_ts)
for rain_delta in [20.0, 30.0, 40.0, 50.0]:
    p4 = base_pred.copy()
    p4[rain_mask] += rain_delta
    r4 = score_rmse(p4)
    print(f"4. July Rain (non-TS) +{rain_delta:.0f}s (n={rain_mask.sum()}): RMSE {r4:.4f} (delta {r4 - baseline_rmse:+.4f} s)")

# 5. Test January Heavy Fog (non-frost) on uncovered flights
fog_mask = is_jan & unc & is_heavy_fog & (~is_frost)
for fog_delta in [20.0, 30.0, 40.0]:
    p5 = base_pred.copy()
    p5[fog_mask] += fog_delta
    r5 = score_rmse(p5)
    print(f"5. Jan Heavy Fog (non-frost) +{fog_delta:.0f}s (n={fog_mask.sum()}): RMSE {r5:.4f} (delta {r5 - baseline_rmse:+.4f} s)")

# 6. Combined Test: Frost (calibrated) + July TS (+150s) + July Rain (+30s) + Jan Fog (+30s)
p_comb = base_pred.copy()
# calibrated frost
for ap, delta in [("LFPG", 40.0), ("EDDF", 30.0), ("EHAM", 50.0), ("LSZH", 35.0), ("LTFM", 35.0), ("LEMD", 40.0)]:
    m = frost_mask & (df["ap"].to_numpy() == ap)
    p_comb[m] += delta
p_comb[ts_mask] += 150.0
p_comb[rain_mask] += 30.0
p_comb[fog_mask] += 30.0
r_comb = score_rmse(p_comb)
print(f"\n6. Combined Package: RMSE {r_comb:.4f} (delta {r_comb - baseline_rmse:+.4f} s vs baseline, {r_comb - r1:+.4f} s vs v49-rule)")
