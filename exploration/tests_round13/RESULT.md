# Round 13 — result: FAIL (both)

Pre-registration: `PREREGISTERED.md`. All variants are applied to the realistic-4 base (v59
weights, with held-out CatBoost); leave one month out over 7 months.

| test | pooled | months better | best 5 rows removed | verdict |
|---|---:|---:|---|---|
| Stage 2, C (trained on the realistic-4 base) vs B (round-12 base, as in v61) | 266.056 → 266.034 (−0.022) | 4/7 | 266.027 → 266.014 | **FAIL** (needs 5/7) |
| LFPG, L1 (realistic-4 base) vs L0 (equal base, as in v56) | 279.510 → 279.513 (+0.004) | 2/6 | worse | **FAIL** |

**Base RMSE on the same rows:** equal 301.94, round-12 (r3) 280.46, realistic-4 (r4) 279.62.

The v61 training base (v42 weights) already captures the useful part of the match. Moving to the
exact v59 weighting, CatBoost included, does not add a reliable gain. v61's correctors stay.
