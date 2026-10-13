#!/usr/bin/env python3
"""Build fearless_barn_v50.parquet according to PREREGISTERED_v50.md:
- Base: fearless_barn_v49.parquet (board score 271.9440)
- Target: 428 July 2026 uncovered departures under active thunderstorm (TS)
- Adjustment: +200.0 s, clamped to [60, 86400], int32
- Exactly 344,413 rows remain 100% byte-identical to v49 (including all January rows).
"""
import json
import sys
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

lines = []
def say(s):
    print(s, flush=True)
    lines.append(s)

say("=== Building fearless_barn_v50.parquet ===")

# 1. Load v49 and v42 base submissions
v49_path = SUB / "fearless_barn_v49.parquet"
v42_path = SUB / "fearless_barn_v42.parquet"
v49 = pd.read_parquet(v49_path)
v42 = pd.read_parquet(v42_path)
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

assert len(v49) == len(tmpl)
assert (v49[config.ID] == tmpl[config.ID]).all()
assert (v42[config.ID] == tmpl[config.ID]).all()

# 2. Identify uncovered flights in July
_, meta = D.load_ranking_features("with_aobt")
assert (meta.index == tmpl[config.ID]).all()

mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month
is_jul = (month.to_numpy() == 7)
is_jan = (month.to_numpy() == 1)

# In July, uncovered flights are those where v49 == v42 (since v49 only modified January)
uncovered_jul = is_jul & (v49[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
say(f"Total rows: {len(v49)}")
say(f"July uncovered flights: {uncovered_jul.sum()}")

# 3. Load & parse METAR for thunderstorm (TS)
wx_frames = []
for st in sorted(adep.unique()):
    fp = WX / f"{st}.csv"
    d = pd.read_csv(fp, dtype=str)
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    wx = d["wxcodes"].fillna("").str.upper()
    sub_df = pd.DataFrame({
        "valid": d["valid"],
        "station": st,
        "is_ts": wx.str.contains("TS", regex=True),
    }).dropna(subset=["valid"]).sort_values("valid")
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

is_ts = merged["is_ts"].fillna(False).to_numpy(dtype=bool)
target_mask = uncovered_jul & is_ts

say(f"Target July thunderstorm uncovered rows: {target_mask.sum()}")
assert target_mask.sum() == 428, f"Expected 428 rows, got {target_mask.sum()}"

# Airport breakdown
say("Breakdown by airport:")
for ap, cnt in adep[target_mask].value_counts().items():
    say(f"  {ap:6s}: {cnt:4d}")

# 4. Apply +200.0s adjustment
v50_sub = v49.copy()
pred = v49[config.TARGET].to_numpy().astype(np.float64)
pred[target_mask] += 200.0
v50_sub[config.TARGET] = np.clip(np.rint(pred), 60, 86400).astype("int32")

# 5. Verify changes
diff = (v50_sub[config.TARGET] - v49[config.TARGET]).to_numpy()
changed = (diff != 0)
say(f"\nVerification:")
say(f"  Total changed rows: {changed.sum()} (must be 428)")
assert changed.sum() == 428
assert (diff[changed] == 200).all()
assert (diff[~changed] == 0).all()

# Check January rows are 100% byte-identical to v49
jan_diff = diff[is_jan]
say(f"  January rows modified: {(jan_diff != 0).sum()} (must be 0)")
assert (jan_diff == 0).all()

# 6. Validate format with submit.validate
errs = submit.validate(v50_sub)
if errs:
    say(f"ERROR: Submission validation failed: {errs}")
    sys.exit(1)
say("  [PASS] submit.validate checks passed.")

# 7. Save submission file and meta
v50_path = SUB / "fearless_barn_v50.parquet"
v50_sub.to_parquet(v50_path, index=False)
say(f"\nSuccessfully wrote: {v50_path}")

meta_json = {
    "arm": "v49_plus_july_convective_thunderstorm",
    "base_file": "submissions/fearless_barn_v49.parquet",
    "version": 50,
    "team": "fearless_barn",
    "n_rows": int(len(v50_sub)),
    "target_rows_adjusted": int(target_mask.sum()),
    "adjustment_sec": 200.0,
    "pred_median": float(v50_sub[config.TARGET].median()),
    "pred_mean": float(v50_sub[config.TARGET].mean()),
    "pre_registered_doc": "tests_weather_metar/PREREGISTERED_v50.md",
}
(SUB / "fearless_barn_v50.parquet.meta.json").write_text(json.dumps(meta_json, indent=2))
(Path(__file__).parent / "build_report_v50.txt").write_text("\n".join(lines) + "\n")
say("Build complete.")
