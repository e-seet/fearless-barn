# PRC Data Challenge 2026 — Taxi-out time prediction

Predict `TAXITIME_SEC_mvt` (seconds from off-block to takeoff) for departures
at ten major European airports. EUROCONTROL PRC Data Challenge 2026, team
**fearless-barn**.

**Best leaderboard score: v61, RMSE 269.9888** (2026-10-07). **Final round:** `exploration/submissions/fearless-barn_final.parquet` (Jan, Feb, Jun, Jul 2026), built with the final design (write-up §12).
A per-airport blend of LightGBM trees, MLPs, FT-Transformers and CatBoost (TabNet at Rome), corrected with
each departure's adsb.lol ADS-B ground track and small weather offsets. Without ADS-B the same
base model scores 286.5256 (v42).

- **Method write-up: [`exploration/WRITEUP.md`](exploration/WRITEUP.md)** (design, validation, use of the leaderboard, results, reproducibility, data licences)
- One-page summary of everything tried: [`exploration/EXPLORATION_SUMMARY.md`](exploration/EXPLORATION_SUMMARY.md)
- Reproduce or predict new data in one command: [`exploration/final_pipeline.py`](exploration/final_pipeline.py)

The sections below describe the repository as it was mid-build; `WRITEUP.md` is the current reference.

## Where to start

| Path | What it is |
|---|---|
| **[`exploration/`](exploration/)** | **The real build.** ~35 candidate pipelines, an evidence-tiered review of every one, and the winning blend. Start with [`exploration/PIPELINE_REVIEW.md`](exploration/PIPELINE_REVIEW.md) (bottom line: what's OK to trust, what's over-tuned and quarantined) then [`exploration/TAXI_OUT_DRIVERS.md`](exploration/TAXI_OUT_DRIVERS.md) (the full hypothesis catalog — every feature/driver tried, most closed negative, with the measurement behind each verdict) |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | Exploratory analysis: taxi-time distribution, anomalous values, missing data, monthly patterns, main airports/runways/aircraft types |
| [`solution/`](solution/) | **Early, superseded baseline** — the first-pass two-arm build (audit, time-based split, naive AOBT-proxy baseline, one LightGBM arm). Superseded in full by `exploration/`; kept for the initial leakage/split audit, which still holds |
| `tools/ml_common/` | Shared checks the pipeline code imports (seeding, split-disjointness, leak controls, triage) — vendored, not hand-rolled per pipeline |
| `prc-2026-datasets/`, `prc-2026-datasets-wx/` | Raw challenge parquets and external METAR weather pulls (both git-ignored, not committed) |

## Getting the data

Not shipped in the repo. `prc-2026-datasets/` (challenge data) and
`prc-2026-datasets-wx/` (external weather, used in the closed weather
experiments) are both git-ignored — challenge data access is credentialed and
team-scoped (MinIO bucket, per-team keys in a local `credentials.json`, itself
git-ignored). See [`exploration/plans/13_live_board_next_steps.md`](exploration/plans/13_live_board_next_steps.md)
§5 for the exact `mc` commands (fetch, upload, score-check).

## Timestamp / target-leakage check

Whether any timestamp field (`AOBT_3_flt`, `EOBT_1_flt`, `IOBT_flt`,
`LOBT_flt`, …) directly or indirectly reconstructs the target is audited in
`stage1_audit.py` ("AOBT_3 near-target proxy" check) and `stage1b_label_quality.py`
(present in both `solution/` and `exploration/` — the check predates the split
and was carried forward unchanged), not in the EDA notebook — kept separate so
the notebook stays a plain look at the data.
