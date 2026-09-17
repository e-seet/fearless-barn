> **RETRACTED (2026-09-17), before any training was run.** The follow-up
> check below falsifies this diagnostic's core assumption. Kept for the
> record with the correction inline rather than deleted, per this project's
> convention (`TAXI_OUT_DRIVERS.md`'s §2l/§2o corrections do the same). Do
> not build the label-correction experiment this file originally
> recommended — see "RETRACTION" at the end.

# Stand-turnover physical check — result

**Question.** LIRF's AOBT-unmatched population (no NM flight record,
~1,488 rows/year — §2q, 0.077% of val rows but 20.68% of total SSE) contains
rows where `BLOCK_TIME_UTC_mvt` sits close to `SCHED_TIME_UTC_mvt` while
takeoff is hours later, producing a huge recorded `TAXITIME_SEC_mvt`. Three
prior approaches (blend-weight solver, OpenAir-proxy cross-check, convex
mix — all §2q) tried to *re-predict* this population better and all failed
out-of-sample. None asked a purely physical question: a stand can only hold
one aircraft at a time, so if another movement is recorded at the SAME
(airport, stand) strictly inside a row's claimed occupancy window
`[BLOCK_TIME, takeoff)`, that is a **deterministic, model-free contradiction**
— the claimed `BLOCK_TIME` cannot be correct, independent of any fitted
weight. Feasibility (checked first): `STAND_mvt` is 99.99%+ populated on
LIRF movements, including this subgroup.

**Setup.** `tests_stand_turnover/run_stand_turnover.py`. Built the full LIRF
movement stream (both DEP and ARR phases, whichever airport column applies)
with `STAND_mvt` + `MVT_TIME_UTC_mvt`, sorted per stand. For each LIRF DEP row
with no flight record and `TAXITIME_SEC_mvt > 3600s` (745 such rows with a
usable stand+block-time in the full training pool; 224 of those fall in
`forward`'s held-out val+test), searched for any OTHER movement (different
`MVT_ID_mvt`) at the same stand with an event time strictly inside
`[BLOCK_TIME, takeoff)`.

## Result — a real, large, physically-provable label defect

**58.9% of the full-pool population (439/745) and 59.4% of the honest
forward/val+test slice (133/224) have a definitive stand-turnover
contradiction** — consistent between the full pool and the held-out slice,
so this isn't a training-pool-specific artifact.

For contradicted rows, the physical bound is far below the recorded label:
median claimed `TAXITIME_SEC_mvt` is 8,760s (2.4h), and the bound implied by
the first contradicting event (`first_contradiction_at_sec`) averages only
**28.5% of the claimed value** (median 25.5%) — i.e. the true upper bound on
these rows' taxi time is typically 3-4x SMALLER than what's recorded as
ground truth. Only 12 of 439 contradicted rows land near the already-known
86,400s (§2g/§2l) date-rollover pattern — **this is a substantially larger
and mechanistically different population than the 15 rows §2l tested and
closed**, not a rediscovery of that bug.

## Reading — this is a LABEL-quality finding, not a prediction-capping lever

**Important distinction from what was originally proposed.** The window
`[BLOCK_TIME, takeoff)` used here requires `BLOCK_TIME_UTC_mvt`, which is
**blanked on ranking DEP rows** (`BLANKED_ON_RANKING_DEP`) — this check
cannot run at inference time on the real scored rows, so it is NOT a way to
detect-and-cap an implausible prediction at submission time. What it
actually reveals: for the majority of this high-SSE TRAINING subpopulation,
the recorded ground-truth label itself is physically impossible, given
independently-observed stand-reuse evidence elsewhere in the same movement
stream (forward-looking movement-stream evidence is organizer-confirmed fair
game, `conversation.md` sec.1, and no `MVT_ID_mvt` self-matches were used).
The model is currently being trained to reproduce these corrupted labels —
in the population that already carries the largest disproportionate share of
squared error in the whole dataset.

**This is a different lever than everything closed on this population
so far.** §2l tested dropping 15 heuristically-flagged rows and found it
decisively net-negative on test (+5.45s) because removing even a few extreme
rows shifted the per-airport blend weight broadly, including at airports
with zero dropped rows. This finding is ~29x larger (439 vs 15 rows) and
evidenced by a general, provable physical constraint rather than a single
known bug pattern — and unlike §2l's exclusion, the natural fix here is
**correction to a known physical upper bound**, not exclusion, which changes
the failure mode §2l hit (a shared blend-weight shift from removing training
mass) into a different, untested one (bounding a corrupted label toward a
value the training pool still contains, changing the label's magnitude but
not the row's existence).

## Disposition

**Not built.** This is a diagnostic establishing prevalence and physical
validity, not a trained model or a submitted lever. ~~The natural next step
— recommended, not yet executed — is a training-label-correction
experiment...~~ **Superseded by the retraction below before that experiment
was run.**

## RETRACTION (2026-09-17): the "physical bound" is wrong, not the recorded label

Before scoping the label-correction experiment, checked the 439 contradicted
rows against `delta_sched_sec` (`MVT_TIME_UTC_mvt − SCHED_TIME_UTC_mvt`) —
the independent proxy §2q already established tracks true `y` almost exactly
on this population's large-delay rows. Result: **the RECORDED label
(`claimed_sec`) is closer to `delta_sched_sec` than the "physical bound" is,
in 93.8% of the 439 contradicted rows** (mean absolute error 1,157s for the
claimed label vs. 9,643s for the bound — the claimed label is ~8x more
consistent with the independent proxy). The worst-case example makes this
concrete: `mvt_id 183903219` claims 131,167s; `delta_sched_sec` independently
says 131,163s (agreement to within 4 seconds); the "physical bound" this
diagnostic computed was 18,239s — wrong by over 100,000s.

**Conclusion: the stand-turnover "contradiction" does not indicate a bad
label. It indicates the underlying physical assumption was wrong** — most
likely because `STAND_mvt` identity does not guarantee exclusive, continuous
occupancy the way this check assumed (remote/de-icing holding positions,
stand reassignment during an extended ground stop, or non-instantaneous
stand release during a real multi-hour delay are all more consistent with
39% of the year's genuinely enormous ground delays landing at LIRF than a
systematic labeling bug at that scale would be). The recorded labels for
this population are corroborated, not contradicted, by independent evidence.

**This also means the evaluation-surface argument applies with full force,
not just as a general caution**: the organizer explicitly does not clean or
filter this data (`conversation.md` sec.2, "kept as-is... will probably be
scored in the ranking too"), and since the recorded values are themselves
well-corroborated, training toward the computed "bound" would have pushed
predictions toward values the scorer would mark wrong on rows where the
recorded label is very likely the more correct one.

**Closed, decisively, before any model was trained.** No compute spent on
the label-correction experiment beyond building the (unrun) harness and this
one cheap proxy check. `TAXI_OUT_DRIVERS.md` §2ae corrected to match. Full
contradicted-row table (now understood as false positives, not true
contradictions): `tests_stand_turnover/contradicted_rows.csv`.
