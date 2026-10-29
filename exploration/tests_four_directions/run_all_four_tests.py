#!/usr/bin/env python3
"""Rigorous Empirical Evaluation of the 4 Unexplored Directions on 2025 Seasonal Data (344,419 rows):

1. Direction 1: Runway-Pair De-icing Routing at LFPG (Paris CDG North vs South)
2. Direction 2: Carrier-Specific De-icing Operations (Home Carriers vs Secondary)
3. Direction 3: Active Runway Crossing Holds at EHAM / LFPG
4. Direction 4: Systematic Ensemble Underprediction Affine Recalibration (Out-of-Sample Transfer)
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

print("Loading seasonal_full test data (Jan + Jul 2025)...")
X_test, y_test, meta_test = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
raw_test = cio.load_training(columns=[
    "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", 
    "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt"
]).set_index("MVT_ID_mvt")

t_test = pd.to_datetime(raw_test["MVT_TIME_UTC_mvt"].reindex(X_test.index), utc=True)
df = raw_test.reindex(X_test.index).copy()
df["y"] = np.asarray(y_test, float)
df["valid"] = t_test.to_numpy()
df["month"] = t_test.dt.month.to_numpy()

# Load base model predictions
tree_sf = np.load(CACHE / "tree_seasonal_full_test.npy")
mlp_sf = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt_sf = np.mean([np.load(EXP / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
base = (tree_sf + mlp_sf + ftt_sf) / 3.0
df["base"] = base
df["resid"] = df["y"] - df["base"]

# Filter out non-finite targets
df = df[np.isfinite(df["y"])].copy()

# Identify ADS-B covered flights
days = sorted(f.stem for f in (EXP / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in ("2025-01", "2025-07"))
e = pd.concat([adsb_features.load_day(d) for d in days]).sort_values("liftoff_dt")
e = e[~e.index.duplicated()]
e.index = e.index.astype("int64")

A7 = ["EDDF", "EHAM", "LEBL", "EDDM", "LSZH", "LEMD", "EGLL"]
df = df.join(e[["e_first"]], how="left")
df["covered"] = df.e_first.notna() & df.ADEP_mvt.isin(A7)

# Parse METAR weather
def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")

wx_frames = []
for st in sorted(df.ADEP_mvt.unique()):
    fp = WX / f"{st}.csv"
    d = pd.read_csv(fp, dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    wx = d["wxcodes"].fillna("").str.upper()
    sub_df = pd.DataFrame({
        "valid": d["valid"],
        "station": st,
        "tmpf": num(d["tmpf"]),
        "dwpf": num(d["dwpf"]),
        "is_snow": wx.str.contains("SN|SG|IC|PL", regex=True),
        "is_fz": wx.str.contains("FZ", regex=True),
        "is_ts": wx.str.contains("TS", regex=True),
    }).dropna(subset=["valid", "tmpf"]).sort_values("valid")
    wx_frames.append(sub_df)

wx_all = pd.concat(wx_frames, ignore_index=True).sort_values("valid")

left = df[["valid", "ADEP_mvt"]].rename(columns={"ADEP_mvt": "station"}).sort_values("valid")
left["mvt_id"] = df.index
merged = pd.merge_asof(
    left,
    wx_all,
    on="valid",
    by="station",
    direction="backward",
    tolerance=pd.Timedelta("90min"),
).set_index("mvt_id").reindex(df.index)

df["tmpf"] = merged["tmpf"]
df["dwpf"] = merged["dwpf"]
df["spread"] = df["tmpf"] - df["dwpf"]
df["is_frost"] = ((df["tmpf"] <= 34.0) & (df["spread"] <= 3.0)).fillna(False)

print("\n" + "="*80)
print("DIRECTION 1: RUNWAY-PAIR DE-ICING ROUTING AT LFPG (PARIS CDG NORTH VS SOUTH)")
print("="*80)
# At LFPG:
# North runway complex: 09L, 09R, 27L, 27R
# South runway complex: 08L, 08R, 26L, 26R
lfpg_jan = df[(df.ADEP_mvt == "LFPG") & (df.month == 1)].copy()
lfpg_frost = lfpg_jan[lfpg_jan.is_frost].copy()
lfpg_normal = lfpg_jan[~lfpg_jan.is_frost].copy()

def classify_lfpg_rwy(r):
    r = str(r).upper().strip()
    if any(k in r for k in ["09L", "09R", "27L", "27R"]): return "North (09/27)"
    if any(k in r for k in ["08L", "08R", "26L", "26R"]): return "South (08/26)"
    return "Other"

lfpg_frost["rwy_pair"] = lfpg_frost["RUNWAY_mvt"].apply(classify_lfpg_rwy)
lfpg_normal["rwy_pair"] = lfpg_normal["RUNWAY_mvt"].apply(classify_lfpg_rwy)

print(f"Total January LFPG frost departures: {len(lfpg_frost)}")
for pair in ["North (09/27)", "South (08/26)"]:
    sub_f = lfpg_frost[lfpg_frost.rwy_pair == pair]
    sub_n = lfpg_normal[lfpg_normal.rwy_pair == pair]
    f_inband = sub_f[sub_f.resid.abs() < 3600]
    n_inband = sub_n[sub_n.resid.abs() < 3600]
    delta = f_inband.resid.mean() - n_inband.resid.mean()
    print(f"\n{pair} Runway Complex:")
    print(f"  Frost count: {len(sub_f)} | Normal count: {len(sub_n)}")
    print(f"  Frost resid mean (in-band): {f_inband.resid.mean():+6.2f} s | median: {f_inband.resid.median():+6.2f} s")
    print(f"  Normal resid mean (in-band):{n_inband.resid.mean():+6.2f} s | median: {n_inband.resid.median():+6.2f} s")
    print(f"  --> Frost Taxi-out Extension (Delta): {delta:+6.2f} s")

print("\n" + "="*80)
print("DIRECTION 2: CARRIER-SPECIFIC DE-ICING (HOME CARRIERS VS SECONDARY)")
print("="*80)
# Top operator by airport (anonymized hash)
for ap, name in [("LFPG", "Primary Hub Carrier"), ("LTFM", "Primary Hub Carrier"), ("EDDF", "Primary Hub Carrier"), ("EGLL", "Primary Hub Carrier")]:
    sub_ap = df[(df.ADEP_mvt == ap) & (df.month == 1) & df.is_frost].copy()
    if len(sub_ap) < 10:
        continue
    top_code = sub_ap.AIRCRAFT_OPERATOR_flt.value_counts().index[0]
    is_home = (sub_ap.AIRCRAFT_OPERATOR_flt == top_code)
    home = sub_ap[is_home]
    sec = sub_ap[~is_home]
    
    h_inband = home[home.resid.abs() < 3600]
    s_inband = sec[sec.resid.abs() < 3600]
    print(f"\n{ap} - {name} ({top_code[:8]}...) vs Secondary Carriers on Frost Days:")
    print(f"  Primary Carrier:    n={len(home):4d} | resid mean: {h_inband.resid.mean():+6.2f} s | median: {h_inband.resid.median():+6.2f} s")
    print(f"  Secondary (Others): n={len(sec):4d} | resid mean: {s_inband.resid.mean():+6.2f} s | median: {s_inband.resid.median():+6.2f} s")
    print(f"  --> Difference (Primary minus Secondary): {h_inband.resid.mean() - s_inband.resid.mean():+6.2f} s")

print("\n" + "="*80)
print("DIRECTION 3: ACTIVE RUNWAY CROSSING HOLDS AT EHAM / LFPG")
print("="*80)
eham_dep = df[df.ADEP_mvt == "EHAM"].copy()
print(f"Total EHAM departures in test set: {len(eham_dep)}")
polder_dep = eham_dep[eham_dep.RUNWAY_mvt.astype(str).str.contains("36L|18R", regex=True)]
other_dep = eham_dep[~eham_dep.RUNWAY_mvt.astype(str).str.contains("36L|18R", regex=True)]
print(f"Polderbaan (36L/18R) departures (crosses active 18C/36C): {len(polder_dep)}")
print(f"  Polderbaan mean resid: {polder_dep[polder_dep.resid.abs()<3600].resid.mean():+6.2f} s | median: {polder_dep[polder_dep.resid.abs()<3600].resid.median():+6.2f} s")
print(f"  Other runways mean resid: {other_dep[other_dep.resid.abs()<3600].resid.mean():+6.2f} s | median: {other_dep[other_dep.resid.abs()<3600].resid.median():+6.2f} s")

print("\n" + "="*80)
print("DIRECTION 4: SYSTEMATIC ENSEMBLE UNDERPREDICTION AFFINE CALIBRATION")
print("="*80)
print("Loading forward_inner validation to fit honest out-of-sample airport bias...")
X_fwd_val, y_fwd_val, m_fwd_val = D.load_part("forward_inner", "val", "with_aobt", filter_target=False)
y_fv = np.asarray(y_fwd_val, float)
valid_fv = np.isfinite(y_fv)
ap_fv = np.asarray(m_fwd_val["ADEP_mvt"]).astype(str)

tree_fv = np.load(CACHE / "tree_forward_inner_val.npy")
mlp_fv = np.load(CACHE / "mlp6_forward_inner_val.npy")
ftt_fv = np.mean([np.load(EXP / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
base_fv = (tree_fv + mlp_fv + ftt_fv) / 3.0
err_fv = y_fv - base_fv

ap_shifts = {}
print("Fitted out-of-sample mean bias per airport (Jul-Aug 2025):")
for ap in sorted(np.unique(ap_fv)):
    m = valid_fv & (ap_fv == ap) & (np.abs(err_fv) < 3600)
    shift = float(np.mean(err_fv[m]))
    ap_shifts[ap] = shift
    print(f"  {ap:6s}: {shift:+6.2f} s")

raw_test_rmse = np.sqrt(np.mean((df.y - df.base)**2))
print(f"\nRaw Base RMSE on Jan+Jul 2025: {raw_test_rmse:.4f} s")

for shrink in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    p_adj = np.array(df.base.to_numpy(), copy=True)
    for ap, shift in ap_shifts.items():
        m = (df.ADEP_mvt == ap).to_numpy()
        p_adj[m] += shrink * shift
    rmse_transferred = np.sqrt(np.mean((df.y.to_numpy() - p_adj)**2))
    print(f"Shrinkage {shrink:.1f} -> Transferred Out-of-Sample RMSE: {rmse_transferred:.4f} s (delta: {rmse_transferred - raw_test_rmse:+.4f} s)")
