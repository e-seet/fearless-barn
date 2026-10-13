#!/usr/bin/env python3
"""Build fearless_barn_v51.parquet according to PREREGISTERED_v51.md:
Weather Refinements on Uncovered Flights:
1. Airport-specific Jan Frost Calibration:
   - LFPG: +15s (making total +45s de-icing offset)
   - LEMD: +30s (making total +60s de-icing offset)
   - EHAM: +50s (making total +80s de-icing offset)
   - EGLL & EDDM: -30s (reverting v49 offset to 0s based on empirical negative/zero delta)
2. Severe Freezing Precipitation (FZ) in Jan: +80s
3. Non-frost Winter Fog / LVP Hold in Jan: +25s (vsby < 1.0 or FG/BR)
4. Uncovered July Rain: +35s (RA/DZ on non-TS flights)

Base: fearless_barn_v50.parquet (Score: 271.8748)
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "solution"
WX = ROOT / "prc-2026-datasets-wx"
SUB = EXP / "submissions"
sys.path.insert(0, str(EXP))
import config
from common import dataset as D
import submit

lines = []
def say(s):
    print(s, flush=True)
    lines.append(s)

say("=== Building fearless_barn_v51.parquet ===")

# 1. Load v50 and v42 base submissions
v50_path = SUB / "fearless_barn_v50.parquet"
v48_path = SUB / "fearless_barn_v48.parquet"
v42_path = SUB / "fearless_barn_v42.parquet"
v50 = pd.read_parquet(v50_path)
v48 = pd.read_parquet(v48_path)
v42 = pd.read_parquet(v42_path)
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

assert len(v50) == len(tmpl)
assert (v50[config.ID] == tmpl[config.ID]).all()
assert (v48[config.ID] == tmpl[config.ID]).all()
assert (v42[config.ID] == tmpl[config.ID]).all()

# Uncovered flights are those without ADS-B corrections in v48
uncovered = (v48[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
say(f"Total rows: {len(v50)}")
say(f"Total uncovered flights: {uncovered.sum()} ({uncovered.mean():.1%})")

# 2. Load ranking metadata
_, meta = D.load_ranking_features("with_aobt")
assert (meta.index == tmpl[config.ID]).all()

mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month
is_jan = (month.to_numpy() == 1)
is_jul = (month.to_numpy() == 7)

# 3. Load & parse METAR
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

for col in ["tmpf", "dwpf", "sknt", "vsby", "gust", "is_snow", "is_fz", "is_rain", "is_ts", "is_fog"]:
    if col in ["tmpf", "dwpf", "sknt", "vsby", "gust"]:
        merged[col] = num(merged[col])
    else:
        merged[col] = merged[col].fillna(False).astype(bool)

merged["spread"] = merged["tmpf"] - merged["dwpf"]
merged["is_frost"] = ((merged["tmpf"] <= 34.0) & (merged["spread"] <= 3.0)).fillna(False).astype(bool)

# Build v51 predictions from v50
pred_v51 = v50[config.TARGET].to_numpy().astype(float)

# 1. Airport calibration on Jan frost
m_frost = is_jan & uncovered & merged["is_frost"]
for ap in np.unique(adep):
    m_ap = m_frost & (adep == ap)
    cnt = m_ap.sum()
    if cnt == 0: continue
    if ap == "LFPG":
        pred_v51[m_ap] += 15.0
        say(f"LFPG frost (+15s beyond v49): {cnt} flights")
    elif ap == "LEMD":
        pred_v51[m_ap] += 30.0
        say(f"LEMD frost (+30s beyond v49): {cnt} flights")
    elif ap == "EHAM":
        pred_v51[m_ap] += 50.0
        say(f"EHAM frost (+50s beyond v49): {cnt} flights")
    elif ap in ["EGLL", "EDDM"]:
        pred_v51[m_ap] -= 30.0
        say(f"{ap} frost (-30s reverting v49): {cnt} flights")

# 2. Freezing precipitation boost (FZ) in Jan
m_fz = is_jan & uncovered & merged["is_fz"]
say(f"Freezing precipitation (FZ) in Jan (+80s): {m_fz.sum()} flights")
if m_fz.sum() > 0:
    pred_v51[m_fz] += 80.0

# 3. Dense Fog / LVP hold in Jan on non-frost flights
m_fog = is_jan & uncovered & (~merged["is_frost"]) & (merged["is_fog"] | (merged["vsby"] < 1.0))
say(f"Non-frost fog flights in Jan (+25s): {m_fog.sum()} flights")
pred_v51[m_fog] += 25.0

# 4. July rain on non-TS flights
m_rain = is_jul & uncovered & (~merged["is_ts"]) & merged["is_rain"]
say(f"Non-TS rain flights in Jul (+35s): {m_rain.sum()} flights")
pred_v51[m_rain] += 35.0

# Round to integer seconds (consistent with submit.py)
pred_v51_int = np.rint(pred_v51).astype("int32")

# Summary of differences
diff = pred_v51_int - v50[config.TARGET].to_numpy()
changed = (diff != 0)
say(f"Total flights modified vs v50: {changed.sum()} ({changed.mean():.2%})")
say(f"Mean shift on modified flights: {diff[changed].mean():+.2f} s")
say(f"Min shift: {diff[changed].min():+d} s, Max shift: {diff[changed].max():+d} s")
say(f"RMSE distance from v50: {np.sqrt(np.mean(diff**2)):.4f} s")

# Assemble and validate dataframe
v51_df = pd.DataFrame({
    config.ID: tmpl[config.ID],
    config.TARGET: pred_v51_int,
})

# Run submit.py validation assertions
submit.validate(v51_df)

# Save output
v51_path = SUB / "fearless_barn_v51.parquet"
v51_df.to_parquet(v51_path, index=False)
say(f"Wrote {v51_path} ({v51_path.stat().st_size / 1e6:.2f} MB)")

# Save sidecar metadata
meta_json = {
    "version": 51,
    "parent_version": 50,
    "date": "2026-10-01",
    "base_score": 271.8748,
    "strategy": "weather_metar_multicomponent_refinement",
    "description": "v50 + calibrated frost offsets by airport (LFPG +45s, LEMD +60s, EHAM +80s, EGLL/EDDM 0s) + Jan freezing precip boost (+80s) + Jan non-frost fog hold (+25s) + Jul non-TS rain hold (+35s)",
    "n_rows_total": len(v51_df),
    "n_rows_modified": int(changed.sum()),
    "offline_validation_2025_delta": -0.0909,
    "expected_board_score": "271.72 to 271.82",
}
(SUB / "fearless_barn_v51.parquet.meta.json").write_text(json.dumps(meta_json, indent=2))
say(f"Wrote {v51_path.name}.meta.json")
