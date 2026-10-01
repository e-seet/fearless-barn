#!/usr/bin/env python3
"""Check LTFM taxi-out times during thunderstorms across all 2025 training data.
Compare TAXITIME_SEC_mvt on TS days/hours vs non-TS days/hours.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "exploration"
WX = ROOT / "prc-2026-datasets-wx"
DATA = ROOT / "prc-2026-datasets"

print("Loading 2025 LTFM departures...")
dfs = []
for p in sorted(DATA.glob("training_*.parquet")):
    d = pd.read_parquet(p, columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt", "TAXITIME_SEC_mvt"])
    sub = d[(d.ADEP_mvt == "LTFM") & (d.PHASE_mvt == "DEP") & d.TAXITIME_SEC_mvt.notna()]
    dfs.append(sub)

ltfm = pd.concat(dfs, ignore_index=True)
ltfm["valid"] = pd.to_datetime(ltfm["MVT_TIME_UTC_mvt"], utc=True)
print(f"Total LTFM 2025 departures: {len(ltfm)}")

# Load METAR
wx = pd.read_csv(WX / "LTFM.csv", dtype=str)
wx["valid"] = pd.to_datetime(wx["valid"], utc=True, errors="coerce")
wx["is_ts"] = wx["wxcodes"].fillna("").str.upper().str.contains("TS", regex=True)
wx["is_rain"] = wx["wxcodes"].fillna("").str.upper().str.contains("RA|DZ", regex=True)
wx = wx.dropna(subset=["valid"]).sort_values("valid")

merged = pd.merge_asof(
    ltfm.sort_values("valid"),
    wx[["valid", "is_ts", "is_rain"]],
    on="valid",
    direction="backward",
    tolerance=pd.Timedelta("90min"),
)

ts = merged.is_ts.fillna(False).to_numpy()
rain = (merged.is_rain.fillna(False) & ~ts).to_numpy()
norm = (~ts & ~rain)

y_ts = merged.TAXITIME_SEC_mvt[ts]
y_rain = merged.TAXITIME_SEC_mvt[rain]
y_norm = merged.TAXITIME_SEC_mvt[norm]

print("\n--- LTFM 2025 FULL YEAR TAXI TIME (SECONDS) ---")
print(f"Thunderstorm (TS) (n={len(y_ts)}): mean {y_ts.mean():.1f} s, median {y_ts.median():.1f} s, p75 {y_ts.quantile(0.75):.1f} s, p90 {y_ts.quantile(0.90):.1f} s")
print(f"Rain (non-TS)     (n={len(y_rain)}): mean {y_rain.mean():.1f} s, median {y_rain.median():.1f} s, p75 {y_rain.quantile(0.75):.1f} s, p90 {y_rain.quantile(0.90):.1f} s")
print(f"Normal            (n={len(y_norm)}): mean {y_norm.mean():.1f} s, median {y_norm.median():.1f} s, p75 {y_norm.quantile(0.75):.1f} s, p90 {y_norm.quantile(0.90):.1f} s")
print(f"\nTS delta vs Normal: mean {y_ts.mean() - y_norm.mean():+.1f} s, median {y_ts.median() - y_norm.median():+.1f} s")
