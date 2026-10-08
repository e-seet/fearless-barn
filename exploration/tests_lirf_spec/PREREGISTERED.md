# LIRF no-record specialist — pre-registration

Written 2026-10-08, before any model is fitted. Board: **v61 = 269.9888** (best). Time-boxed: decision by
2026-10-09 18:00.

**Idea and credit.** A separate model for LIRF departures without an NM flight record (`FLIGHT_ID_mvt`
missing), predicting the residual from the schedule gap. Published by team quirky-honey
(github.com/dhsdb010/OPDI, `specialist.py`, GPL-3.0), who credit team zestful-fountain
(Phoenix-Ops-LTD/prc2026-taxiout). quirky-honey report board 302.01 → 277.91 from it. Our implementation is
our own, from their written description.

**Scope.** LIRF, no NM record: 397 rows in Jan + Jul 2025; 1,017 in the 7 test months; 383 on the ranking set.
Our held-out recipe F has RMSE 5,304 s there on Jan + Jul (28 % of the squared error).

**Model S.**
- CatBoost (depth 5, 1,000 trees, lr 0.04, l2 10, 5 seeds averaged), on scope rows only, with our 57
  `with_aobt` features.
- Target y − g with g = max(0, MVT − SCHED); prediction g + residual, floored at 30 s, capped at
  max(g, 88,500 s).
- For each test month, trained on scope rows of the other 11 months of 2025.

**Reference.** F (realistic-4 base + K2 stage-2 corrector), clipped to the LOBT window. Both arms are
clipped, so the reference carries every fixed rule the candidate carries (the v63 lesson).

**Comparison.** On the 7 months Jan, Jul–Dec 2025, whole-month RMSE with scope rows replaced by S.

**PASS (all required):**
1. pooled RMSE better;
2. Jan + Jul pooled better;
3. better in at least 5 of 7 months;
4. still better with the 20 largest-gain rows removed.

**Build if PASS:** v64 = v61 with the 383 ranking scope rows replaced by S (trained on all 2025 scope
rows), then the LOBT clip. The expectation is written before upload; if v64 scores above 269.9888, v61 is
re-uploaded last.
