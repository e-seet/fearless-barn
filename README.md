# PRC Data Challenge 2026 — Taxi-out time prediction

Predict `TAXITIME_SEC_mvt` (seconds from off-block to takeoff) for departures
at ten major European airports. EUROCONTROL PRC Data Challenge 2026, team
**fearless-barn**.

**Final result: RMSE 295.21 on the live leaderboard** (submission v25), rank
35/108 at the time it was scored. Model: a per-airport blend of a LightGBM
tree family and an 18-seed-averaged MLP, refit on the full 12-month training
period. Full writeup and evidence trail: **[`exploration/PIPELINE_REVIEW.md`](exploration/PIPELINE_REVIEW.md)**
(what's trustworthy and why) and **[`exploration/plans/13_live_board_next_steps.md`](exploration/plans/13_live_board_next_steps.md)**
(the actual submission-by-submission board history).

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
