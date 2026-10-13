# ADS-B round 3 (v45 candidate) — pre-registration

Written 2026-09-30, after v43 (281.8629) and v44 (279.1517, best and last), **before any Sep–Oct
2025 ADS-B day is fetched**.

**Board lesson:** both uploads beat even the naive projection. The corrector underestimates itself
locally, so more ADS-B coverage is the lever.

## Fresh surface: Sep–Oct 2025

- **Days:** all 61 days of Sep–Oct 2025.
- **Base:** the forward-split val predictions, from models trained on Jan–Aug only:
  `(tree_forward_val + mlp12_forward_val + ftt seed42 forward pv) / 3`.
- No ADS-B test has touched these months.

## Question 1: a bigger training set for the corrector

- **R44:** the v44 corrector as saved (`model_v44`, Jan + Jul 2025 only).
- **R45:** the same features, airports and parameters, trained on Jan + Jul 2025 (held-out
  seasonal_full base) **plus Nov + Dec 2025** (held-out forward-test base).
- **R45 is adopted only if** its pooled RMSE on Sep–Oct is lower than R44's, and it wins on at least
  half the days.

## Question 2: more airports (LSZH, LEMD, EGLL), each on its own

A corrector for airport X is trained on X's covered rows in Jan + Jul + Nov + Dec 2025, frozen, and
scored on Sep–Oct. **X is added only if:**
- X's RMSE on Sep–Oct improves;
- at least 2/3 of the days improve at X;
- it beats the best of 5 permutation controls at X.

This relaxes the earlier requirement that X improve in both Jan ↔ Jul directions, and the reason is
recorded now:
- LEMD has no Jan 2025 coverage, so that check cannot be run for it;
- a fresh season scored by a frozen model is the stronger test.

Sep–Oct has never been used to choose anything.

**If any airport is added:** the combined corrector (EDDF, EHAM, LEBL, EDDM and the added airports,
trained on Jan + Jul + Nov + Dec, or on Jan + Jul if R45 is not adopted) must beat the chosen
4-airport corrector on Sep–Oct pooled.

## Build

- **v45** = v42 plus the winning corrector.
- The expectation is written before upload.
- It is uploaded only if it differs from v44.
- If it scores worse than 279.1517, v44 is re-uploaded last as a byte copy.

**Still excluded:**
- LIRF: 2026 Jul has 0 % coverage, and the corrector hurt LIRF on Jan → Jul;
- LFPG and LTFM: almost no coverage.
