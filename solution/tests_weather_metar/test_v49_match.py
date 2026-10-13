#!/usr/bin/env python3
"""Check the exact row matching for v49 pre-registration:
- January 2026 departures
- v48.TAXITIME_SEC_mvt == v42.TAXITIME_SEC_mvt (uncovered by ADS-B)
- METAR at departure time (as-of join backward <= 90min) with tmpf <= 34.0 and (tmpf - dwpf) <= 3.0
Expected count: exactly 4,878 rows (LFPG 2,223, EDDF 1,257, LSZH 377, EHAM 311, LTFM 286, EDDM 242, others 182).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "solution"
WX = ROOT / "prc-2026-datasets-wx"
sys.path.insert(0, str(EXP))
import config
from common import dataset as D

# Load ranking set & base submissions
v48 = pd.read_parquet(EXP / "submissions" / "fearless_barn_v48.parquet")
v42 = pd.read_parquet(EXP / "submissions" / "fearless_barn_v42.parquet")
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

assert (v48[config.ID] == tmpl[config.ID]).all()
assert (v42[config.ID] == tmpl[config.ID]).all()

# Load ranking metadata for timestamps and ADEP
_, meta = D.load_ranking_features("with_aobt")
assert (meta.index == tmpl[config.ID]).all()

mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month

uncovered = (v48[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
is_jan = (month.to_numpy() == 1)

print(f"Total ranking rows: {len(tmpl)}")
print(f"January rows: {is_jan.sum()}")
print(f"Uncovered rows total: {uncovered.sum()}")
print(f"Uncovered January rows: {(uncovered & is_jan).sum()}")

# Parse METAR for all airports
def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")

wx_frames = []
for st in sorted(meta["ADEP_mvt"].unique()):
    fp = WX / f"{st}.csv"
    d = pd.read_csv(fp, dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    tmpf = num(d["tmpf"])
    dwpf = num(d["dwpf"])
    sub_df = pd.DataFrame({
        "valid": d["valid"],
        "station": st,
        "tmpf": tmpf,
        "dwpf": dwpf,
    }).dropna(subset=["valid", "tmpf", "dwpf"]).sort_values("valid")
    wx_frames.append(sub_df)

wx_all = pd.concat(wx_frames, ignore_index=True).sort_values("valid")

# Prepare ranking queries for as-of join
left = pd.DataFrame({
    "mvt_id": tmpl[config.ID].to_numpy(),
    "valid": mvt_time.to_numpy(),
    "station": adep.to_numpy(),
    "uncovered": uncovered,
    "is_jan": is_jan,
}).sort_values("valid")

merged = pd.merge_asof(
    left,
    wx_all,
    on="valid",
    by="station",
    direction="backward",
    tolerance=pd.Timedelta("90min"),
).set_index("mvt_id").reindex(tmpl[config.ID])

print(f"Matched METAR rows in ranking: {merged['tmpf'].notna().sum()} / {len(merged)}")

frost = (merged["tmpf"] <= 34.0) & ((merged["tmpf"] - merged["dwpf"]) <= 3.0)
target_rows = is_jan & uncovered & frost.to_numpy()

print(f"\nTotal target rows (is_jan & uncovered & frost): {target_rows.sum()}")
print("\nBreakdown by airport:")
for ap, cnt in adep[target_rows].value_counts().items():
    print(f"  {ap}: {cnt}")
