# ADS-B corrector at LIRF, flights with an NM record only — pre-registration

Written 2026-10-02, before any model is fitted. Board: v50 = 271.8748 (best and last, #50/217).

## Why this differs from the rejected LIRF test (round 4)

The round-4 LIRF corrector was trained and scored on **all** LIRF rows, including the no-record
departures whose extreme labels are record artifacts. ADS-B cannot see those. This test restricts
both training and correction to LIRF departures **with** an NM flight record (`FLIGHT_ID_mvt`
present; 98–99 % of LIRF 2026 rows) **and** a strict-matched ADS-B ground track.

## Base at LIRF (as shipped)

The shipped LIRF prediction is 0.541 × MLP + 0.459 × TabNet; the FTT weight is 0. The held-out bases
mirror it:

| months | MLP | TabNet |
|---|---|---|
| Jul | `mlp6_seasonal_full_test` | mean of `preds_seed{42,43,44}_sf_lr2e-3p` `pt` |
| Aug | `mlp18_forward_inner_val` | mean of `build/inner_seed{42,43,44}` `pv` |
| Sep–Oct | `mlp12_forward_val` | mean of `preds_seed{42,43,44}_lr2e-3p` `pv` |
| Nov–Dec | `mlp12_forward_test` | mean of `preds_seed{42,43,44}_lr2e-3p` `pt` |

## Corrector

- The v48 design: the same 7 ADS-B inputs plus base and hour, the v46 parameters, seeds 42–46
  averaged.
- **LIRF only.**
- Trained on LIRF rows with a record and a track.

## Surface

- Leave one month out over **Jul, Aug, Sep, Oct, Nov, Dec 2025**. Jan 2025 has no LIRF ADS-B.
- Training uses the other 5 months.
- Scored on **all LIRF rows** of the held-out month; only rows with a record and a track are changed.

## PASS requires all of:

1. LIRF RMSE pooled over the 6 months improves;
2. it improves in at least 4 of the 6 months;
3. it still improves with the 5 rows of largest gain removed;
4. it beats the best of 3 permutation controls (ADS-B inputs shuffled among the eligible rows).

## Known limits, stated now

- It acts on **January 2026 only**: LIRF has 0 % ADS-B in July 2026.
- January is a season the LIRF test cannot cover, because there is no Jan 2025 coverage.
- Expected gain: a few tenths of a second at most.

## Build

**PASS →** v55:
- v50 plus the LIRF corrector (trained on all 6 months), added to v50 on the January 2026 LIRF rows
  with a record and a strict track;
- every other row byte-identical to v50;
- expectation written before upload; if it scores above 271.8748, v50 is re-uploaded last.

**FAIL →** closed.

**Cut-off:** 2026-10-06.
