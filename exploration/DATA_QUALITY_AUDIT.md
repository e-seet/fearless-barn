# Data-quality audit: train2025 vs. ranking.parquet

A systematic pass over missingness, duplication, and identifier consistency,
comparing the training population against the actual scored (ranking) rows.
Motivated by `EXTERNAL_PLAN_CROSSCHECK.md` item 14 ("no single systematic
table has been built") and the v27 menu. Restricted throughout to `PHASE_mvt
== "DEP"` rows at the 10 target airports, since that's the scored
population. Script: `scratchpad/data_quality_audit.py` (this session).

Verdict key: **Known** (already documented/handled elsewhere in this repo,
re-confirmed here, not a new finding) · **New** (not previously quantified)
· **Caveat** (a raw number in this audit that looks alarming but is a
measurement artifact, not a real issue — flagged so it isn't miscited later).

---

## 1. Phase and airport coverage — Known, clean

DEP/ARR split is ~50/50 in both train (2,085,047 / 2,082,750) and ranking
(344,841 / 344,693). **100% of DEP rows in both train and ranking belong to
the 10 target airports** — the 1,107 distinct `ADEP_mvt` values seen in the
raw ranking.parquet file earlier this session (including odd codes like
`KPIE`, `CYSJ`) all come from ARR-phase rows, where `ADEP_mvt` is the
*origin* of an inbound flight and can legitimately be anywhere in the world.
Confirms the existing `PHASE_mvt.eq("DEP")` filtering in `common/dataset.py`
is doing exactly what it should — no stray-airport contamination in the
scored population.

## 2. Missingness — Known pattern, New magnitude

All `_flt`-suffixed columns (`CALLSIGN_flt`, `ADEP_flt`, `AOBT_3_flt`, etc.)
are null together, at the same rate as `FLIGHT_ID_mvt` — this is the
documented "no matching flight record" join-failure population, not
independent per-column missingness. `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt` are exactly 100% null in ranking DEP rows and 0% null in
train — the `BLANKED_ON_RANKING_DEP` contract, confirmed working exactly as
specified, nothing else blanked.

**New: the join-failure rate itself is meaningfully higher in the scored
population than in training.**

| | train | ranking | relative increase |
|---|---:|---:|---:|
| `_flt`-column null rate | 1.078% | 1.534% | **+42%** |
| LIRF `AOBT_3_flt` null rate | 0.926% (1,488/160,704) | 1.424% (383/26,899) | **+54%** |

This wasn't previously quantified. The LIRF number refines and confirms the
session's existing "~1,488 unmatched rows/year" figure (exact match on the
train side) but adds the new observation that the *scored* population has a
proportionally larger unmatched slice. Since this population is already
known to be the model's worst-performing segment (RMSE ~4,039s on this
subgroup vs. ~250-300s pooled, from the earlier OpenAir-override
cross-check), a higher prevalence at scoring time is a plausible **partial
explanation for the large, roughly-constant local-to-board gap every arm in
this family carries** (~47-51s for both v25 and v26) — not a defect in any
one blend, but a property of the scored population itself containing more
of the hardest rows than the training population does. Not something a
model can be trained around (the rows are hard because they're missing the
predictive columns), but useful context: it argues the gap is structural
rather than a sign of a specific bug, consistent with this session's
broader finding (§2o) that the 3-way blend's local edge is real and the
gap is best read as board-level effects, not a hidden defect.

## 3. Duplicates — Caveat, not a real issue

`MVT_ID_mvt` (the row identifier) is 100% unique in both train and ranking
— clean. `FLIGHT_ID_mvt` showed 22,426 "duplicated" rows in train (1.076%)
and 5,283 in ranking (1.532%) — **but these fractions match the join-failure
rate almost exactly** (1.078% / 1.534% from section 2). This is a
measurement artifact: `pandas.Series.duplicated()` treats multiple `NaN`
values as duplicates of each other by default, so this is re-counting the
same missing-`FLIGHT_ID_mvt` population, not genuine duplicate flight
records. **Caveat noted so this number isn't miscited as "22,426 duplicate
flights" later** — a proper re-check would exclude nulls first; not done
here since the near-exact match to the known null rate already explains it.

## 4. `_mvt` vs `_flt` identifier consistency — New (small, real effect after correcting for the same artifact)

Same caveat applies to the raw numbers: comparing `X_mvt != X_flt` when
`X_flt` is null evaluates to `True` in pandas, so the raw mismatch rate is
inflated by the same join-failure population. Subtracting that baseline
(1.078% train / 1.534% ranking) from each raw rate gives the **genuine**
identifier-disagreement rate, where both columns are present but disagree:

| comparison | raw mismatch (train / ranking) | genuine mismatch, join-failure subtracted |
|---|---:|---:|
| `ADEP_mvt` vs `ADEP_flt` | 1.078% / 1.534% | **~0%** (fully explained by the join-failure population) |
| `ADES_mvt` vs `ADES_flt` | 2.480% / 2.957% | **~1.40% / ~1.42%** |
| `AIRCRAFT_TYPE_mvt` vs `AIRCRAFT_TYPE_flt` | 1.414% / 1.712% | **~0.34% / ~0.18%** |

`ADEP` disagreement is fully explained by missing joins — no real
departure-airport inconsistency. `ADES` (destination) disagreement is real
and non-trivial (~1.4% of rows, stable between train and ranking): the
movement record's arrival airport and the flight plan's filed destination
genuinely disagree at this rate — plausibly diversions or filed-vs-actual
destination changes. Aircraft-type disagreement is smaller (~0.2-0.3%) but
also real. **Not previously quantified in this repo.** Both are small
enough that they're unlikely to be a meaningful lever (consistent with this
session's broader finding that under-1%-of-rows categorical effects
repeatedly close near-zero — §2h, §2j, §2n's stand-level check), but worth
recording as a genuine, if minor, data-quality characteristic.

## 5. Target validity — Known, all previously documented

388 training rows with taxi-out ≤0s (min -12s) — matches the documented
IMPOSSIBLE-VALUE FLOOR policy (`config.py`), dropped from training
unconditionally. 202 rows >3h, 69 rows >6h — consistent with the
already-characterized heavy tail, nothing new. `BLOCK_TIME_UTC_mvt ==
SCHED_TIME_UTC_mvt` for 8,123 rows (0.39%) — matches the organizer-confirmed
BLOCK≈SCHED artifact documented in `conversation.md`.

## 6. Per-airport coverage — Known, clean

Ranking row counts per airport are a sensible fraction of training volume
(roughly 15-18% each, consistent with ~2 scored months out of a
12-month training year, adjusted for July's 3-airport restriction). No
anomalies.

---

## Summary

Two genuinely new findings, both minor in isolation but the first is
potentially explanatory at the whole-pipeline level:

1. **The join-failure population (missing flight records, including LIRF's
   AOBT-unmatched rows) is proportionally 42-54% larger in the scored
   ranking set than in training.** Since this is the model's worst-handled
   segment, this is a plausible structural contributor to the large,
   roughly-constant local-to-board RMSE gap this entire arm family carries
   — not a defect to fix, but useful context for interpreting future local
   val numbers against board scores.
2. **A small but real (~1.4%) `ADES_mvt`/`ADES_flt` disagreement** and a
   smaller (~0.2-0.3%) aircraft-type disagreement, previously unquantified.
   Given how consistently sub-1%-of-rows categorical effects have closed
   near-zero elsewhere this session, low prior for these being a usable
   lever, but recorded for completeness.

Everything else checked (phase/airport coverage, the `BLANKED_ON_RANKING_DEP`
contract, target validity, the BLOCK≈SCHED artifact, per-airport row
counts) confirms existing documentation with no surprises. One caveat: the
raw "duplicate `FLIGHT_ID_mvt`" and raw `ADEP` mismatch numbers in this
audit's own script output are measurement artifacts of null-handling, not
real duplicate/mismatch counts — noted above so they aren't miscited later.
