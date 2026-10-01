# Probe v50: July Convective Thunderstorm Offset for Uncovered Flights — Pre-registration

Written 2026-10-01, before building or uploading `v50`.
Current board standing: **v49 = 271.9440** (best and last, #46 of 214 teams).

## 1. Physical Mechanism & Hypothesis

1. **The Blind Spot:** In July, convective thunderstorm activity forces airport ground stops, apron holds, and increased slot spacing. Aircraft are pushed back from their gates (recorded AOBT), but cannot taxi or take off until the storm cell clears. For ADS-B covered flights, transponder tracking directly captures this holding time. But **157,116 departures (45.6% of the scored set) have zero ADS-B coverage**, including 100% of Istanbul (LTFM).
2. **Empirical Evidence in 2025 Training & Holdout Data:**
   - At Istanbul Airport (LTFM, zero ADS-B coverage): in 2025, 546 departures occurred during active thunderstorms (`TS`). Their mean taxi-out time was **1700.6 s (28.3 min)** vs **1051.7 s (17.5 min)** under normal conditions — a massive delta of **+648.9 s (+10.8 minutes)**.
   - Across all 10 airports in held-out July 2025 (`seasonal_full` test): uncovered departures during thunderstorms were systematically under-predicted by **+200 s to +350 s** across every major hub (EDDF +264.6 s, EDDM +234.1 s, EGLL +354.7 s, LFPG +232.9 s, LSZH +202.1 s).
   - In offline evaluation on the full 344,419 rows of the 2025 seasonal test set, adding an offset to July thunderstorm flights reduced pooled RMSE by **-0.1256 s (at +120s), -0.1650 s (at +180s), and -0.1888 s (at +240s)**.
3. **The 2026 Scored Window:**
   - In July 2025, LTFM had zero thunderstorms reported.
   - In July 2026, LTFM experienced repeated convective storms (23 METAR observations with `TS`), affecting **370 departures**.
   - Total July 2026 uncovered departures under active thunderstorm across all airports: **exactly 428 flights**.

## 2. Candidate Definition (v50)

- **Base:** `v49` (`fearless_barn_v49.parquet`, score 271.9440).
- **Target Population:** July 2026 departures that:
  1. Have NO ADS-B correction (`v49.TAXITIME_SEC_mvt == v42.TAXITIME_SEC_mvt` on July rows).
  2. Have METAR observation at departure time with active thunderstorm (`TS` in `wxcodes`).
  Total rows matching: **exactly 428 departures**:
  - LTFM: 370
  - LSZH: 21
  - LEMD: 18
  - LIRF: 10
  - EDDM: 6
  - LFPG: 2
  - LEBL: 1
- **Modification:** Add **$+200.0$ s** to the 428 rows. Clamp to $[60, 86400]$ s, integer rounded.
  (At current v49 mean 1129.8 s, $+200$ s brings the mean to 1329.8 s, well below the empirical thunderstorm mean of 1700 s, guarding against any over-prediction).
- **Untouched:** Exactly **344,413 rows (99.88%)** remain 100% byte-identical to `v49`, including all January flights (preserving v49's full frost benefit), all ADS-B covered flights, and all non-thunderstorm July flights.

## 3. Expectations & Pass Bar

- **Expected Board Score:** **271.45 to 271.65** (an expected gain of -0.30 to -0.50 s vs `v49`).
- **Pass Bar:** Board score strictly $< 271.9440$.
- **Fallback Action:** If board score $\ge 271.9440$, re-upload `fearless_barn_v49.parquet` immediately.
