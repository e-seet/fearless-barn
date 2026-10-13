# Probe v49: Winter Morning De-icing Offset for Uncovered Flights — Pre-registration

Written 2026-10-01, before building or uploading `v49`.
Current board standing: **v48 = 272.1714** (best and last).

## Physical Mechanism & Hypothesis

1. **The Blind Spot:** Our base ensemble (`v42`) was trained without weather data. While ADS-B captures de-icing delays directly through `stop_s` and `lineup_s` for covered flights, **156,206 flights (45.3% of the scored set) have zero ADS-B coverage** (including 100% of LFPG and LTFM, and ~55% of EDDF).
2. **Measured Effect in Held-Out January 2025:**
   - At Frankfurt (EDDF) on non-ADS-B flights: frost morning departures ($T \le 1^\circ\text{C}$ and $T - T_d \le 1.7^\circ\text{C}$) are under-predicted by **+42.7 s** compared to non-frost flights.
   - At Paris (LFPG, in-band $y < 3600$ s): frost departures are under-predicted by **+46.4 s** compared to non-frost flights.
   - Across all 10 airports in January 2025: 7 of 8 cold hubs show consistent $+16$ s to $+84$ s under-prediction during frost.
   - Applying a $+30$ s de-icing adjustment to uncovered frost flights lowered January 2025 RMSE by **-0.115 s to -0.144 s**.

## Candidate Definition (v49)

- **Base:** `v48` (`fearless_barn_v48.parquet`).
- **Target Population:** January 2026 departures that:
  1. Have NO ADS-B correction in `v48` (`v48.TAXITIME_SEC_mvt == v42.TAXITIME_SEC_mvt`).
  2. Have METAR observation at departure time with `tmpf <= 34.0` (<= 1.1°C) and `tmpf - dwpf <= 3.0` (<= 1.7°C).
  Total rows matching: **4,878 departures** (LFPG 2,223, EDDF 1,257, LSZH 377, EHAM 311, LTFM 286, EDDM 242, others 182).
- **Modification:** Add $+30$ s to the 4,878 rows. Clamp to $[60, 86400]$ s.
- **Untouched:** All July departures (0% frost), all ADS-B covered departures (188,635 rows), and all non-frost departures are 100% byte-identical to `v48`.

## Expectations & Pass Bar

- **Expected Board Score:** **272.09 to 272.14** (an expected gain of -0.03 to -0.08 s).
- **Pass Bar:** Board score strictly $< 272.1714$.
- **Fallback Action:** If board score $\ge 272.1714$, re-upload `fearless_barn_v48.parquet` immediately to maintain the "last submission counts" rule.
