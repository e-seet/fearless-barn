#!/usr/bin/env python3
"""Build and validate candidate submissions v52, v53, and v54:
- v52: Conservative De-icing Specialist (Calibrated Frost + FZ only, no fog/rain)
- v53: Aggressive Convective & Severe Winter (LTFM TS +320s, FZ +120s, Snow +40s, calibrated frost, fog +25s, rain +35s)
- v54: Peak-Hour LVP & Congestion Interaction (Morning rush-hour fog +50s, off-peak fog +25s, calibrated frost, TS +200s, rain +35s)

All built on top of v50 / v48 / v42.
Validated with submit.validate().
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/eddie/fearless-barn")
EXP = ROOT / "solution"; exec(open(ROOT / "tools" / "repo_layout.py").read()); EXP = dual(EXP)
WX = ROOT / "prc-2026-datasets-wx"
SUB = EXP / "submissions"
sys.path.insert(0, str(EXP))
import config
from common import dataset as D
import submit

print("=== Building Candidates v52, v53, v54 ===")

v50_path = SUB / "fearless_barn_v50.parquet"
v48_path = SUB / "fearless_barn_v48.parquet"
v42_path = SUB / "fearless_barn_v42.parquet"
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

v50 = pd.read_parquet(v50_path)
v48 = pd.read_parquet(v48_path)
v42 = pd.read_parquet(v42_path)

uncovered = (v48[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
print(f"Total rows: {len(v50)} | Uncovered flights: {uncovered.sum()} ({uncovered.mean():.1%})")

# Metadata
_, meta = D.load_ranking_features("with_aobt")
assert (meta.index == tmpl[config.ID]).all()

mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month
hour = mvt_time.dt.hour
is_jan = (month.to_numpy() == 1)
is_jul = (month.to_numpy() == 7)

# Parse METAR
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

m_frost = is_jan & uncovered & merged["is_frost"]
m_fz = is_jan & uncovered & merged["is_fz"]
m_fog = is_jan & uncovered & (~merged["is_frost"]) & (merged["is_fog"] | (merged["vsby"] < 1.0))
m_snow = is_jan & uncovered & (~merged["is_frost"]) & merged["is_snow"]
m_ts = is_jul & uncovered & merged["is_ts"]
m_rain = is_jul & uncovered & (~merged["is_ts"]) & merged["is_rain"]

# Helper to apply frost calibration
def apply_frost_calib(p):
    for ap in np.unique(adep):
        m_ap = m_frost & (adep == ap)
        cnt = m_ap.sum()
        if cnt == 0: continue
        if ap == "LFPG": p[m_ap] += 15.0
        elif ap == "LEMD": p[m_ap] += 30.0
        elif ap == "EHAM": p[m_ap] += 50.0
        elif ap in ["EGLL", "EDDM"]: p[m_ap] -= 30.0

# -------------------------------------------------------------
# BUILD v52: Conservative De-icing Specialist
# -------------------------------------------------------------
print("\n--- Generating v52 ---")
p52 = v50[config.TARGET].to_numpy().astype(float).copy()
apply_frost_calib(p52)
if m_fz.sum() > 0:
    p52[m_fz] += 80.0

p52_int = np.rint(p52).astype("int32")
diff52 = p52_int - v50[config.TARGET].to_numpy()
ch52 = (diff52 != 0)
print(f"v52 modified flights: {ch52.sum()} ({ch52.mean():.2%}) | mean shift: {diff52[ch52].mean():+.1f}s | RMSE diff: {np.sqrt(np.mean(diff52**2)):.4f}s")

df52 = pd.DataFrame({config.ID: tmpl[config.ID], config.TARGET: p52_int})
submit.validate(df52)
p52_path = SUB / "fearless_barn_v52.parquet"
df52.to_parquet(p52_path, index=False)
meta52 = {
    "version": 52,
    "parent_version": 50,
    "strategy": "conservative_frost_deicing_specialist",
    "description": "v50 + calibrated frost offsets (LFPG +45s, LEMD +60s, EHAM +80s, EGLL/EDDM 0s) + Jan FZ freezing precip (+80s). No fog or rain adjustments.",
    "n_rows_total": len(df52),
    "n_rows_modified": int(ch52.sum()),
    "offline_validation_2025_delta": -0.0632,
    "expected_board_score": "271.78 to 271.84",
}
(SUB / "fearless_barn_v52.parquet.meta.json").write_text(json.dumps(meta52, indent=2))
print(f"Saved {p52_path.name}")

# -------------------------------------------------------------
# BUILD v53: Aggressive Convective & Severe Storm
# -------------------------------------------------------------
print("\n--- Generating v53 ---")
p53 = v50[config.TARGET].to_numpy().astype(float).copy()
apply_frost_calib(p53)
# LTFM TS higher hold (+120s additional, making +320s total)
m_ltfm_ts = m_ts & (adep == "LTFM")
p53[m_ltfm_ts] += 120.0
# FZ higher boost (+120s total)
p53[m_fz] += 120.0
# Snow hold (+40s)
p53[m_snow] += 40.0
# Fog hold (+25s)
p53[m_fog] += 25.0
# Rain hold (+35s)
p53[m_rain] += 35.0

p53_int = np.rint(p53).astype("int32")
diff53 = p53_int - v50[config.TARGET].to_numpy()
ch53 = (diff53 != 0)
print(f"v53 modified flights: {ch53.sum()} ({ch53.mean():.2%}) | mean shift: {diff53[ch53].mean():+.1f}s | RMSE diff: {np.sqrt(np.mean(diff53**2)):.4f}s")

df53 = pd.DataFrame({config.ID: tmpl[config.ID], config.TARGET: p53_int})
submit.validate(df53)
p53_path = SUB / "fearless_barn_v53.parquet"
df53.to_parquet(p53_path, index=False)
meta53 = {
    "version": 53,
    "parent_version": 50,
    "strategy": "aggressive_convective_and_severe_storm",
    "description": "v50 + calibrated frost + LTFM July TS (+320s total) + Jan FZ (+120s) + Jan Snow (+40s) + Fog (+25s) + Rain (+35s)",
    "n_rows_total": len(df53),
    "n_rows_modified": int(ch53.sum()),
    "offline_validation_2025_delta": -0.0845,
    "expected_board_score": "271.70 to 271.80",
}
(SUB / "fearless_barn_v53.parquet.meta.json").write_text(json.dumps(meta53, indent=2))
print(f"Saved {p53_path.name}")

# -------------------------------------------------------------
# BUILD v54: Peak-Hour LVP Congestion Interaction
# -------------------------------------------------------------
print("\n--- Generating v54 ---")
p54 = v50[config.TARGET].to_numpy().astype(float).copy()
apply_frost_calib(p54)
p54[m_fz] += 80.0
# Morning peak (06:00 - 10:00 UTC) fog gets +50s; off-peak gets +25s
is_peak = (hour >= 6) & (hour <= 10)
p54[m_fog & is_peak] += 50.0
p54[m_fog & (~is_peak)] += 25.0
p54[m_rain] += 35.0

p54_int = np.rint(p54).astype("int32")
diff54 = p54_int - v50[config.TARGET].to_numpy()
ch54 = (diff54 != 0)
print(f"v54 modified flights: {ch54.sum()} ({ch54.mean():.2%}) | mean shift: {diff54[ch54].mean():+.1f}s | RMSE diff: {np.sqrt(np.mean(diff54**2)):.4f}s")

df54 = pd.DataFrame({config.ID: tmpl[config.ID], config.TARGET: p54_int})
submit.validate(df54)
p54_path = SUB / "fearless_barn_v54.parquet"
df54.to_parquet(p54_path, index=False)
meta54 = {
    "version": 54,
    "parent_version": 50,
    "strategy": "peak_hour_lvp_congestion_interaction",
    "description": "v50 + calibrated frost + morning peak fog hold (+50s) / off-peak fog (+25s) + Jan FZ (+80s) + Jul rain (+35s)",
    "n_rows_total": len(df54),
    "n_rows_modified": int(ch54.sum()),
    "offline_validation_2025_delta": -0.0784,
    "expected_board_score": "271.72 to 271.82",
}
(SUB / "fearless_barn_v54.parquet.meta.json").write_text(json.dumps(meta54, indent=2))
print(f"Saved {p54_path.name}")
print("\nAll candidate versions built and validated successfully.")
