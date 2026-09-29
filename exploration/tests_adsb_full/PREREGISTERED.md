# ADS-B (adsb.lol) full test — pre-registration

Written 2026-09-30, before any model is fitted on ADS-B features. Base: v42 = 286.5256 (best and last).

## Why a new test

`tests_adsb_coverage` kept one crude feature on 3 days and fitted a straight line. That test only
showed low coverage, so it could not close the idea. This test does it properly.

## What has been seen already (exploratory, `check_2025.py`, `check_strict.py`)

- **Strict matching:** one aircraft per departure, and its ground-to-air change must fall within
  ±120 s of the recorded takeoff.
- **Coverage:** 2025-01-15 47 %, 2025-07-15 44 %, 2026-01-15 62 %. It varies strongly by day and
  airport; LFPG and LTFM are near 0.
- **The pushback estimate is late:** the first ground sighting comes 4–11 min after the recorded
  off-block at most airports, and within about 1–1.5 min at EDDM only.
- **Best-case single-slope fix,** fitted in-sample on covered rows (MLP residual): Jan 236 → 231,
  Jul 278 → 276. Expected pooled effect: about −1 to −2 s in Jan and < 0.5 s in Jul, before any
  out-of-sample loss.

**Expectation, written now:** likely FAIL, or a gain under 1 s pooled.

## Data

- 18 days of 2025, extracted by `fetch.py`:
  - January: 06 09 13 15 18 21 24 28 31;
  - July: 03 07 10 15 17 20 23 26 30.
- Features per departure from `features.py` (strict matching):
  - `e_first`, `e_disp`, `e_move`;
  - `n_ground`, `liftoff_dt`;
  - `covered` (0/1).
- Uncovered rows have the features missing.

## Base predictions (held out: seasonal_full was trained without Jan and Jul)

On the seasonal_full test rows of these days:
- `base = (tree + mlp6 + ftt_small) / 3`, where `ftt_small` is the mean of the lr1e-3 seasonal_full
  seeds in `tests_ftt`;
- at LIRF, `base = 0.541 · mlp6 + 0.459 · tabnet_sf` (the shipped LIRF recipe).

## Method (fixed)

- A LightGBM corrector on the residual `r = y − base`.
- It is trained on **covered rows only** and applied to covered rows. Uncovered rows keep `base`.
- Inputs: the ADS-B features above, `base`, airport (categorical), and hour of the takeoff.
- Parameters: `objective=l2`, `learning_rate=0.03`, `n_estimators=300`, `num_leaves=15`,
  `min_data_in_leaf=50`, `feature_fraction=0.8`, `seed=42`.
- **Cross-month in both directions:** train on the January days and evaluate on the July days, and
  the reverse.
- **Permutation control:** the same corrector with the ADS-B features shuffled among covered rows
  (within airport), 5 shuffles. This measures how much a corrector gains from `base`, airport and
  hour alone.

## Bar

PASS requires all of:
1. The corrector lowers **pooled RMSE over all rows** of the evaluation days, in **both** directions
   (Jan → Jul and Jul → Jan).
2. In both directions the gain beats the **best** of the 5 permutation controls.
3. The pooled gain, averaged over the two directions, is at least **0.3 s**.

**PASS →**
- fetch Jan and Jul 2026 (the rows that are scored) and more 2025 training days;
- rebuild the corrector on all fetched 2025 days;
- **no upload until the user has settled the rules question** (whether a flight's own ADS-B ground
  track is allowed as an input). One board probe after that, with v42 as the fallback.

**FAIL →** closed as a *full* test, and the records are updated.

## Addendum (2026-09-30, before the full run): airport-focused variant

Added at the user's request ("focus on the months or airports with a lot of issues"). It was
written after the 2-day smoke run, which had 5 % of rows covered and is not informative per
airport, and before the full run.

**Rule:**
- the airports that get the correction are the ones where the Jan → Jul corrector lowers that
  airport's RMSE on July;
- the correction is then applied **only at those airports** and judged on Jul → Jan (months not
  used to choose the airports);
- it is also run the other way: choose the airports on Jul → Jan, judge on Jan → Jul.

**PASS (secondary):**
- the pooled gain from the chosen airports is negative in both directions, beats the best of the
  5 permutation controls there, and averages at least 0.3 s;
- a secondary PASS leads to the same next steps as the main PASS.

**Known limits:**
- LFPG and LTFM have almost no ADS-B, so this cannot help them;
- LIRF has 0 % coverage on the January 2025 day checked so far, so LIRF cannot be validated across
  months unless the other January days cover it.
