# Round 11: ADS-B corrector capacity, pushed further — pre-registration

Written 2026-10-07, before any model is fitted. Board: **v59 = 270.81** (best and last).

**Why:** corrector capacity paid on the board both times it was raised:
- v46: 600 trees and 31 leaves were adopted;
- v57: 1,500 trees, 63 leaves and lr 0.02 gave −0.68 on the board, 7/7 months locally.

No ceiling has been seen yet. The LIRF and LFPG correctors still use the old 600 / 31 setting.

## Part A: the 7-airport corrector (stage 2)

- **Reference K** (= `model_v57`): 1,500 trees, 63 leaves, lr 0.02, min_data_in_leaf 50,
  feature_fraction 0.8, seeds 42–46.
- **K2:** 3,000 trees, 127 leaves, lr 0.015, everything else as K.
- **K3:** 3,000 trees, 63 leaves, lr 0.01, everything else as K.
- **Surface:** leave one month out over Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025 (`tests_round10`
  frame, held-out bases); scored on all rows.
- **PASS (each against K):**
  - pooled better;
  - better in at least 5 of 7 months;
  - still better with the 5 largest-gain rows removed.
- **If both pass:** the one with the lower pooled RMSE is used, and only that one.

## Part B: the LIRF and LFPG correctors at the K setting

**LIRF (stage 4):**
- the `tests_lirf_adsb` frame: LIRF departures with a record and a track, leave one month out over
  Jul–Dec 2025, scored on all LIRF rows;
- the v55 setting against the K setting (seeds 42–46);
- **PASS:** LIRF RMSE pooled better; better in at least 4 of 6 months; still better with the best
  5 rows removed.

**LFPG (stage 5):**
- the `tests_round9` frame: LFPG rows with a track and a record, leave one month out over 7 months,
  scored on all rows;
- the v56 setting against the K setting;
- **PASS:** pooled better; better in at least 2/3 of the months where it changes rows; still better
  with the best 5 rows removed.

## Build

- **v60** = `final_pipeline.py` on the v59 base, with every passing change; all other stages as in
  v59.
- The expectation is written before upload. If v60 scores above 270.81, v59 is re-uploaded last.
- **If nothing passes:** v59 stays final.

**Cut-off:** built by 2026-10-08 12:00.
