# Round 9 — result

Pre-registration: `PREREGISTERED.md`. Leave one month out over 7 months; A = the v48 corrector
design; pooled RMSE over all rows (A = 290.237).

| test | pooled vs A | months better | best 5 removed | best permutation vs A | verdict |
|---|---:|---:|---|---:|---|
| R: 7-airport corrector on record rows only | +0.261 | 0/7 | worse | +11.68 | **FAIL** |
| **C: LFPG corrector (track + record rows)** | **−0.098** | 6/7 | 290.203 → 290.124 | +0.031 | **PASS** |
| G: EGLL gets its own model | −0.005 | 6/7 | 290.067 → 290.076 (worse) | +0.907 | **FAIL** |

**Notes:**
- **R:** removing the 2,851 tracked no-record rows hurts everywhere. At the 7 airports those rows are
  ordinary flights, unlike LIRF's no-record rows.
- **G:** EGLL's tiny gain is carried by a handful of rows.
- **C:** LFPG has 7,407 eligible rows in Jul–Dec 2025 and none in Jan 2025.

**v56** = v55 plus the LFPG corrector (trained on all 7,407 rows, 5 seeds):
- applied to 2026 LFPG departures with a track and a record: 1,973 of 39,872 (Jan 1,002, Jul 971);
- 1,964 rows change, all of them LFPG;
- correction mean +93.8 s, mean |.| 124.4 s, 1–99 % [−154, +956];
- byte-identical from the saved boosters.

**Expectation written before upload: 271.62–271.81 (central about 271.70)** against v55 = 271.764.
- **Risks:** the large positive mean shift, and January cannot be validated (no Jan 2025 LFPG
  coverage).
- If it scores above 271.764, v55 is re-uploaded last.
