#!/usr/bin/env python3
"""Direction 2: Continuous Physics-Based Runway Queue Simulator.
Unlike previous tests_runway_queue (which used crude 15m/30m window count features in a stage-2 corrector),
this simulator models the discrete-event queue at each runway threshold:
1. Computes unimpeded rolling transit time stand -> runway threshold based on historical median.
2. Estimated Threshold Arrival Time (ETAT) = AOBT + unimpeded_time.
3. Sorts flights at each runway by ETAT.
4. Simulates the departure sequence with wake-vortex separation:
   - Super (A380): requires 150s trailing separation
   - Heavy (B777/A350/B787): requires 120s trailing separation
   - Medium/Light: requires 60s trailing separation
5. Output simulated runway delay: W_sim = departure_slot - ETAT.
6. Evaluates correlation with true taxi-out time and test RMSE impact.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "exploration"
sys.path.insert(0, str(EXP))
import config
from common import dataset as D
from common import io as cio

print("Loading seasonal_full data for queue simulation...")
X_train, y_train, meta_train = D.load_part("seasonal_full", "train", "with_aobt", filter_target=True)
X_test, y_test, meta_test = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)

raw_train = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "WK_TBL_CAT_flt", "RUNWAY_mvt", "STAND_mvt", "ADEP_mvt"]).set_index("MVT_ID_mvt")
raw_test = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "WK_TBL_CAT_flt", "RUNWAY_mvt", "STAND_mvt", "ADEP_mvt"]).set_index("MVT_ID_mvt")

df_train = raw_train.reindex(X_train.index).copy()
df_train["y"] = np.asarray(y_train, float)
df_train["t"] = pd.to_datetime(df_train["MVT_TIME_UTC_mvt"], utc=True)

df_test = raw_test.reindex(X_test.index).copy()
df_test["y"] = np.asarray(y_test, float)
df_test["t"] = pd.to_datetime(df_test["MVT_TIME_UTC_mvt"], utc=True)

print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

# 1. Compute historical unimpeded transit time per ADEP + STAND + RUNWAY (or ADEP + RUNWAY fallback)
stand_rwy_unimpeded = df_train.groupby(["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])["y"].quantile(0.10).to_dict()
ap_rwy_unimpeded = df_train.groupby(["ADEP_mvt", "RUNWAY_mvt"])["y"].quantile(0.10).to_dict()
ap_unimpeded = df_train.groupby("ADEP_mvt")["y"].quantile(0.10).to_dict()

def get_unimpeded(row):
    key3 = (row["ADEP_mvt"], row["STAND_mvt"], row["RUNWAY_mvt"])
    if key3 in stand_rwy_unimpeded and pd.notna(stand_rwy_unimpeded[key3]):
        return stand_rwy_unimpeded[key3]
    key2 = (row["ADEP_mvt"], row["RUNWAY_mvt"])
    if key2 in ap_rwy_unimpeded and pd.notna(ap_rwy_unimpeded[key2]):
        return ap_rwy_unimpeded[key2]
    return ap_unimpeded.get(row["ADEP_mvt"], 600.0)

print("Computing unimpeded transit times...")
df_test["unimpeded"] = [get_unimpeded(row) for _, row in df_test.iterrows()]
df_test["etat"] = df_test["t"] + pd.to_timedelta(df_test["unimpeded"], unit="s")

# 2. Run discrete-event queue simulation per airport and runway
def get_sep(leading_wtc):
    if leading_wtc in ["J", "SUPER"]: return 150.0
    if leading_wtc in ["H", "HEAVY"]: return 120.0
    return 60.0

print("Simulating runway threshold queuing sequence on test set...")
sim_delays = []

for (ap, rwy), group in df_test.groupby(["ADEP_mvt", "RUNWAY_mvt"]):
    sorted_grp = group.sort_values("etat")
    current_avail_time = pd.Timestamp.min.tz_localize("UTC")
    
    for idx, row in sorted_grp.iterrows():
        arr_time = row["etat"]
        wtc = str(row["WK_TBL_CAT_flt"]).upper()
        
        # If threshold is busy, wait until available
        if arr_time < current_avail_time:
            wait = (current_avail_time - arr_time).total_seconds()
            dep_time = current_avail_time
        else:
            wait = 0.0
            dep_time = arr_time
            
        sep = get_sep(wtc)
        current_avail_time = dep_time + pd.Timedelta(seconds=sep)
        
        sim_delays.append((idx, wait))

sim_df = pd.DataFrame(sim_delays, columns=["MVT_ID_mvt", "sim_delay"]).set_index("MVT_ID_mvt")
df_test["sim_delay"] = sim_df.reindex(df_test.index)["sim_delay"].fillna(0.0)

# Check correlation with true target y
valid_mask = np.isfinite(df_test["y"])
y_valid = df_test["y"][valid_mask]
sim_valid = df_test["sim_delay"][valid_mask]
unimp_valid = df_test["unimpeded"][valid_mask]

corr_raw = np.corrcoef(sim_valid, y_valid)[0, 1]
print(f"\nCorrelation of simulated queue delay with true taxi time: rho = {corr_raw:+.4f}")

# Check with baseline predictions
from tests_tail_diagnosis.diagnose import CACHE
tree_sf = np.load(CACHE / "tree_seasonal_full_test.npy")
mlp_sf = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt_sf = np.mean([np.load(EXP / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
base = (tree_sf + mlp_sf + ftt_sf) / 3.0

resid = y_valid.to_numpy() - base[valid_mask]
corr_resid = np.corrcoef(sim_valid, resid)[0, 1]
print(f"Correlation of simulated queue delay with BASE MODEL RESIDUAL: rho = {corr_resid:+.4f}")

# Test if adding alpha * sim_delay reduces base residual RMSE
for alpha in [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]:
    adjusted = base[valid_mask] + alpha * sim_valid.to_numpy()
    rmse = np.sqrt(np.mean((y_valid.to_numpy() - adjusted)**2))
    print(f"Alpha {alpha:.2f} -> RMSE = {rmse:.4f} s (delta: {rmse - np.sqrt(np.mean(resid**2)):+.4f} s)")
