# ADS-B round 4 (v46 candidate) — pre-registration

Written 2026-10-01, after v45 = 273.5977 (best and last; −5.55 vs v44, beating its written
expectation of 275.0–277.5), **before any Aug 2025 ADS-B day is fetched**.

## Fresh surface: Aug 2025

- **Days:** all 31 days of Aug 2025.
- **Base:** the forward_inner val predictions, from models trained on Jan–Jun only:
  `(tree_forward_inner_val + mlp18_forward_inner_val + mean FTT lr1e-3 seeds 42–50 pv) / 3`.
- Only the August rows are scored. July was used for training.
- No ADS-B test has touched August.

## Q1: add Sep–Oct to the training set

- **R45:** the saved `model_v45`, trained on Jan + Jul + Nov + Dec.
- **R46:** the same, plus Sep + Oct (held-out forward-val base).
- **R46 is adopted only if** its Aug pooled RMSE is below R45's, and it wins on at least half the
  days.

## Q2: one capacity variant

- **C:** the Q1 winner's training set with `n_estimators = 600` and `num_leaves = 31`; everything
  else unchanged.
- **Adopted only if** its Aug pooled RMSE is below the Q1 winner's, and it wins on at least half the
  days.
- This is the only hyperparameter tried.

## Q3: LIRF (affects only Jan 2026, where 63 % of LIRF rows have ADS-B; Jul 2026 has 0 %)

A LIRF-only corrector is trained on LIRF covered rows in the Q1 winner's months, frozen, and scored on
Aug. LIRF is heavy-tailed and the corrector hurt LIRF on Jan → Jul before, so the bar is stricter.
**LIRF is added only if all of these hold:**
- LIRF RMSE on Aug improves;
- at least 2/3 of the days improve at LIRF;
- it beats the best of 5 permutation controls;
- it still improves with the 5 rows of largest gain removed.

**If LIRF is added:** the combined corrector (7 airports + LIRF) must also beat the 7-airport
corrector on Aug pooled.

## Build

- **v46** = v42 plus the winning recipe, only if it differs from v45's.
- The expectation is written before upload.
- If it scores worse than 273.5977, v45 is re-uploaded last as a byte copy.
