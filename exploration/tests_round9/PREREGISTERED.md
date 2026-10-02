# Round 9: three ADS-B corrector refinements — pre-registration

Written 2026-10-02, before any model is fitted. Board: **v55 = 271.764** (best and last).

The lesson from v55 (LIRF): restricting the ADS-B corrector to departures **with an NM flight record**
made it work. No-record rows carry record artifacts that ADS-B cannot see.

## Reference A (all three tests compare against it)

- The v48 corrector design: 7 airports (EDDF, EHAM, LEBL, EDDM, LSZH, LEMD, EGLL), the v46 inputs and
  parameters, seeds 42–46 averaged.
- Trained and applied on rows with a strict ADS-B track.
- **Leave one month out** over Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025 (the same held-out bases as
  rounds 5–7).
- Scored on **all rows** of the held-out month.

## Test 1 (R): record rows only, at the 7 airports

The same as A, but training **and** correction use only tracked rows **with** `FLIGHT_ID_mvt`. Tracked
no-record rows keep the base.

## Test 2 (C): an LFPG corrector

A, plus an LFPG-only corrector (v48 design, seeds 42–46):
- trained on LFPG rows with a track and a record from the other 6 months;
- applied to LFPG rows with a track and a record.

Months where LFPG has no eligible rows change nothing.

## Test 3 (G): EGLL gets its own model

The same as A, except EGLL rows are corrected by an EGLL-only model (v48 design, trained on EGLL
tracked rows) instead of the shared one. EGLL is named now because its tracked rows keep the highest
corrected error (RMSE 260 against 125–200 elsewhere).

## Bars (each test on its own, against A)

1. pooled RMSE better than A;
2. better in at least 5 of 7 months. For C: in at least 2/3 of the months where it changes any row;
3. still better with the 5 rows of largest gain removed;
4. better pooled than the best of 3 permutation controls (ADS-B inputs shuffled among the rows that
   the test changes).

## Build

If any test passes, **v56** combines the passing ones:

    v56 = v55 + (new ADS-B stage − v48 ADS-B stage)

- The new ADS-B stage is the passing variant(s), trained on all 7 months, applied to v42.
- The weather-offset rows and the LIRF correction stay exactly as in v55.
- If R and G both pass, they are combined (G's EGLL model trained on record rows) only if the
  combination beats each of them pooled in the same LOMO run. Otherwise the better single one is used.

The expectation is written before upload. If v56 scores above 271.764, v55 is re-uploaded last.

**If nothing passes:** v55 stays final.
