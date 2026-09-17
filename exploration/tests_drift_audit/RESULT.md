# Feature-distribution drift audit — result

**Question.** `DATA_QUALITY_AUDIT.md` covered missingness, duplicates, and
identifier consistency between train2025 and the real `ranking.parquet`, but
not a systematic per-feature numeric-distribution comparison. An outside
competitor (`conversation.md` sec.4) reported checking "all 52 numeric
features, 2025 January+July against the 2026 ranking set... none shifted
more than 0.25 sd." No persisted artifact in this repo confirms that scope
was ever replicated here — `DATA_QUALITY_AUDIT.md` names its own script as
`scratchpad/data_quality_audit.py`, which doesn't survive across sessions.
This closes that gap rather than re-litigating a result that was never
actually on record.

**Setup.** `tests_drift_audit/run_drift_audit.py`. Built `with_aobt` features
for the full train2025 pool and the real `ranking.parquet` (both restricted
to `PHASE_mvt=="DEP"` at the 10 target airports, matching
`DATA_QUALITY_AUDIT.md`'s scope). 42 numeric feature columns compared (every
column not in `F.CATEGORICAL`). Two comparisons, both standardized by the
train column's own SD, threshold |z| > 0.25 (matching the community's
reported figure so results are comparable):

1. **Full training pool vs. each scored ranking month separately** (Jan
   2026, Jul 2026 — Jul restricted to the 3 airports the real ranking file
   covers, matching `config.SPLITS["seasonal"]`). The actual train/serve
   population gap.
2. **Same calendar month, year-over-year** (train's own Jan/Jul 2025 vs.
   ranking's Jan/Jul 2026, airport-matched). Isolates genuine year-boundary
   drift by removing the seasonal-composition confound (1) can't separate.

## Result — clean on the decisive comparison, expected drift on the other

**Comparison 2 (same-month, year-over-year): zero features flagged, on both
Jan and Jul.** No numeric feature in this build shows meaningful year-over-
year drift when compared like-for-like by calendar month. This directly
addresses the one caveat `TAXI_OUT_DRIVERS.md` §2o flagged as genuinely
open (a year-boundary effect between Dec 2025 and Jan 2026 that no
within-2025 holdout could represent) — at the feature-distribution level,
there is no such effect, at least not one visible in these 42 columns.

**Comparison 1 (full pool vs. scored month): 9 of 42 flagged for January,
2 of 42 for July — all explainable, none alarming.**

| feature | Jan z | Jul z | reading |
|---|---:|---:|---|
| `month_num`, `doy`, `doy_sin`, `doy_cos` | up to −1.68 | up to −1.30 | Trivial: a single calendar month's day-of-year distribution differs from the full 12-month pool's by construction. Exactly why `config.SPLITS["seasonal"]` exists. |
| `arr_prev_60m`, `arr_prev_30m`, `arr_prev_15m`, `dep_prev_60m`, `dep_next_60m` | −0.25 to −0.29 | (not flagged) | January-only, small, all in the same direction (fewer nearby movements than the yearly average) — consistent with January being a genuinely quieter traffic month, not a bug. Barely over threshold and only on the congestion-count family. |

No blanked-column leak, no unexpected feature reading dramatically different
values, no surprise columns. The July-only flags are purely the same
calendar-concentration artifact as January's, smaller because July's
3-airport-restricted composition happens to sit closer to the pool average.

## Disposition

**Clean.** No mechanical drift bug found. This does not explain the
~47-51s local-to-board RMSE offset (`DATA_QUALITY_AUDIT.md`'s join-failure-
rate finding remains the only identified partial explanation) — it rules out
a second, independent explanation (a feature-distribution bug) rather than
finding one. Bug-detector, not a lever, as scoped going in: this buys
confidence for a future submission, it does not itself improve the score.
Full data: `tests_drift_audit/drift_summary.csv`.
