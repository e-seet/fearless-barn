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

## Addendum 2 (2026-09-30, before the full run): movement features and the 2025 → 2026 drift gate

This follows from re-reading the forum thread the user pasted.
- GREKI: "adsb.lol's receiver network changed between the years, so features encoding where an
  aircraft was heard didn't transfer; features describing its movement did."
- arnavhm13 (2026 coverage): MAD good, CDG thin, IST none, FCO none in July 2026.

Several MAIN features are reception-type (`n_ground`, `e_first`, `liftoff_dt`, and partly `e_disp`
and `e_move`). A Jan ↔ Jul 2025 test cannot see a 2025 → 2026 shift, so it is fixed now:

1. **Movement features** (`features.MOVE`, all measured from the first displacement from the stand):
   - `path_km`: taxi path length;
   - `stop_s` and `n_stops`: time stopped and number of stops (< 2 kt for ≥ 30 s);
   - `lineup_s`: time from the end of the last stop to liftoff;
   - `med_gs`: median taxi speed.
2. **Variants,** each judged by the same bar and permutation control:
   - `main` (as registered above);
   - `move` (MOVE only);
   - `all` (MAIN + MOVE).
   Reporting all three is fixed now, so no variant can be chosen quietly.
3. **Drift gate** (`drift.py`):
   - Jan 2025 days are compared with Jan 2026 days, and Jul 2025 with Jul 2026 (at least 3 days of
     2026 per month: Jan 15/20/27 and Jul 8/15/22);
   - a feature is *drifting* if its pooled standardized mean difference exceeds 0.2 in either month;
   - a variant that passes the Jan ↔ Jul bar may go forward **only if none of its features drift**;
   - otherwise it is re-run without the drifting features, and that re-run must pass the same bar.
4. Everything before an upload is unchanged: the full 2026 fetch, and the user's decision on the
   rules. The moderator said "open data sources can be usable", and the question about a flight's
   own ground track was asked on the forum but never answered in what we have.

## Addendum 3 (2026-09-30, after the first full run and the drift report, before the drift-clean re-run)

**First full run** (18 days): `main`, `move` and `all` all PASS, and all three focused variants
PASS. Mean pooled gain: −1.09, −0.72 and −1.29 s for the main variants; −2.77, −1.96 and −3.06 s for
the focused ones.

**Drift report:**
- `n_ground` drifts in Jan and Jul (|SMD| 0.21 and 0.37);
- `path_km` and `n_stops` drift in Jul (0.40 and 0.36);
- every other feature has |SMD| ≤ 0.16.

Coverage also moved a lot between the years:
- July: EDDM 35 → 96 %, EGLL 11 → 96 %, LIRF 64 → 0 %;
- January: LEMD 0 → 97 %, LIRF 0 → 52 %.

**The drift rule applies:** each variant is re-run without its drifting features.

| variant | features |
|---|---|
| `main_nd` | `e_first`, `e_disp`, `e_move`, `liftoff_dt` |
| `move_nd` | `stop_s`, `lineup_s`, `med_gs` |
| `all_nd` | `main_nd` + `move_nd` |

The same bar and the same focused rule apply to each.

**Fixed now, for the build that follows a PASS:**
- the variant carried forward is the drift-clean variant with the largest mean *focused* gain, if it
  passes;
- the correction is applied only at the airports chosen in **both** directions (the intersection);
- LFPG, LTFM and LIRF are never corrected: LFPG and LTFM have almost no coverage, and LIRF has no
  2025 January coverage to validate on;
- expected board effect: well under the local pooled gain. The local base is 338 against 286 on the
  board, and the 2026 coverage mix differs; the expectation is written before any upload.

## Addendum 4 (2026-09-30): the shipped configuration, and confirmation on fresh days

**Shipped configuration** (`ship_config.py`, 18 days; scored after addendum 3 fixed the rule):
- variant `all_nd`, applied only at EDDF, EHAM and LEBL;
- Jan → Jul: −2.71 s pooled, 9/9 days better, −2.65 s with the best row removed;
- Jul → Jan: −4.89 s pooled, 9/9 days better, −4.78 s with the best row removed.

Those three airports were chosen on these same 18 days, and this is the best of several drift-clean
variants, so the 18-day numbers are biased upward.

**Confirmation** (`confirm.py`, fixed now, before any of these days is fetched):
- Fresh days: the other 44 days of Jan and Jul 2025 (`fresh_days.txt`).
- The configuration is frozen. Only the corrector's training rows change, and they are the old
  days only:
  - trained on the 9 old January days, scored on the fresh July days;
  - trained on the 9 old July days, scored on the fresh January days.
- **PASS requires:**
  - pooled RMSE over all rows of the fresh days improves in both directions;
  - it beats the best of 5 permutation controls in both directions;
  - at least 2/3 of the fresh days improve in each direction.

**PASS →**
- fetch the remaining Jan and Jul 2026 days;
- build the final corrector on all 62 Jan + Jul 2025 days;
- write the board expectation: roughly −0.5 to −1.5 s. The local base is weaker (338 vs 286), and
  the 2026 coverage mix differs;
- no upload until the user decides the rules point.

**FAIL →** ADS-B closed, with this full record.
