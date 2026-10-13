#!/usr/bin/env python3
"""Build fearless_barn_v49.parquet according to PREREGISTERED_v49.md:
- Base: fearless_barn_v48.parquet (board score 272.1714)
- Target: 4,878 January 2026 uncovered departures under frost conditions (tmpf <= 34.0, tmpf - dwpf <= 3.0)
- Adjustment: +30 s, clamped to [60, 86400], int32
- Exactly 339,963 rows remain 100% identical to v48.
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

say("=== Building fearless_barn_v49.parquet ===")

# 1. Load v48 and v42 base submissions
v48_path = SUB / "fearless_barn_v48.parquet"
v42_path = SUB / "fearless_barn_v42.parquet"
v48 = pd.read_parquet(v48_path)
v42 = pd.read_parquet(v42_path)
tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)

assert len(v48) == len(tmpl)
assert (v48[config.ID] == tmpl[config.ID]).all()
assert (v42[config.ID] == tmpl[config.ID]).all()

# 2. Identify uncovered flights in v48
uncovered = (v48[config.TARGET].to_numpy() == v42[config.TARGET].to_numpy())
say(f"Total rows: {len(v48)}")
say(f"Uncovered flights in v48: {uncovered.sum()} ({uncovered.mean():.1%})")

# 3. Load ranking metadata for timestamps and airport
_, meta = D.load_ranking_features("with_aobt")
assert (meta.index == tmpl[config.ID]).all()

mvt_time = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
adep = meta["ADEP_mvt"].astype(str)
month = mvt_time.dt.month
is_jan = (month.to_numpy() == 1)

# 4. Load & parse METAR
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

assert merged["tmpf"].notna().all(), "Missing METAR observations!"

frost = (merged["tmpf"] <= 34.0) & ((merged["tmpf"] - merged["dwpf"]) <= 3.0)
target_mask = is_jan & uncovered & frost.to_numpy()

say(f"Target frost uncovered rows in January 2026: {target_mask.sum()}")
assert target_mask.sum() == 4878, f"Expected 4878 rows, got {target_mask.sum()}"

# Breakdown
say("Breakdown by airport:")
for ap, cnt in adep[target_mask].value_counts().items():
    say(f"  {ap:6s}: {cnt:5d}")

# 5. Apply +30s adjustment
v49_sub = v48.copy()
pred = v48[config.TARGET].to_numpy().astype(np.float64)
pred[target_mask] += 30.0
v49_sub[config.TARGET] = np.clip(np.rint(pred), 60, 86400).astype("int32")

# 6. Verify changes
diff = (v49_sub[config.TARGET] - v48[config.TARGET]).to_numpy()
changed = (diff != 0)
say(f"\nVerification:")
say(f"  Total changed rows: {changed.sum()} (must be 4878)")
assert changed.sum() == 4878
assert (diff[changed] == 30).all()
assert (diff[~changed] == 0).all()

# 7. Validate format with submit.validate
errs = submit.validate(v49_sub)
if errs:
    say(f"ERROR: Submission validation failed: {errs}")
    sys.exit(1)
say("  [PASS] submit.validate checks passed.")

# 8. Save submission file and meta
v49_path = SUB / "fearless_barn_v49.parquet"
v49_sub.to_parquet(v49_path, index=False)
say(f"\nSuccessfully wrote: {v49_path}")

meta_json = {
    "arm": "v48_plus_frost_deicing_heuristic",
    "base_file": "submissions/fearless_barn_v48.parquet",
    "version": 49,
    "team": "fearless_barn",
    "n_rows": int(len(v49_sub)),
    "target_rows_adjusted": int(target_mask.sum()),
    "adjustment_sec": 30.0,
    "pred_median": float(v49_sub[config.TARGET].median()),
    "pred_mean": float(v49_sub[config.TARGET].mean()),
    "pre_registered_doc": "tests_weather_metar/PREREGISTERED_v49.md",
}
(SUB / "fearless_barn_v49.parquet.meta.json").write_text(json.dumps(meta_json, indent=2))
(Path(__file__).parent / "build_report_v49.txt").write_text("\n".join(lines) + "\n")
say("Build complete.")
