# Probe v51: Multi-Component Weather Refinements on Uncovered Flights — Pre-registration

**Date:** 2026-10-01  
**Status:** Pre-registered, built, validated, and staged for upload when submission quota resets at 00:00 UTC.  
**Baseline on Public Board:** `fearless-barn_v50.parquet` — **`271.8748`** (Rank #46 / 214 teams)  
**Parent Submission:** `fearless_barn_v50.parquet`  
**Built File:** [`fearless_barn_v51.parquet`](file:///c:/Users/Admin/Desktop/eddie/fearless-barn/solution/submissions/fearless_barn_v51.parquet) (2.22 MB, 344,841 rows)  
**Build Script:** [`build_v51.py`](file:///c:/Users/Admin/Desktop/eddie/fearless-barn/solution/tests_weather_metar/build_v51.py)  

---

## 1. Physical Motivation & Rationale

`v49` (January morning frost de-icing, $+30$s) and `v50` (July convective thunderstorms, $+200$s) established that targeted physical weather adjustments on the **157,116 departures with zero ADS-B coverage** consistently improve the competition score (from `272.1714` down to `271.8748`).

However, `v49` treated all frost conditions identically across all airports and missed other distinct meteorological events:
1. **Airport De-icing Disparities:** Large hub airports with complex de-icing pads and remote taxi routings (e.g. LFPG Paris, LEMD Madrid, EHAM Amsterdam) experience substantially longer holds than smaller layout stations or airports where uncovered flights behave differently (EGLL, EDDM).
2. **Severe Freezing Precipitation (`FZ`):** Freezing rain/drizzle (`FZRA`, `FZDZ`, `FZFG`) creates active ice accumulation requiring Type IV anti-icing fluid and extended holdover times, far beyond simple morning frost.
3. **Low Visibility Procedures (LVP) during Fog (`FG`/`BR` or vsby < 1.0 mi):** Under dense fog, airports enforce Low Visibility Procedures: taxiway centerlines are restricted, runway protection areas are enlarged, and taxi speeds are reduced, adding holding time regardless of temperature.
4. **July Convective Rain (`RA`/`DZ`):** Summer rain showers without full thunder create ramp wetness and taxi spacing buffers.

---

## 2. Empirical Validation on Held-Out 2025 Test Set (`seasonal_full` test)

Evaluated across all 344,419 rows of the held-out 2025 seasonal test set (January + July 2025):

| Model / Candidate | 2025 Test RMSE | $\Delta$ vs `v50` | Description |
|---|---|---|---|
| **Base Model** | 375.9403 s | $+0.2175$ s | No weather adjustments on uncovered flights |
| **Simulated `v49`** | 375.8975 s | $+0.1747$ s | Flat $+30$ s frost on all uncovered January flights |
| **Simulated `v50`** | 375.7228 s | baseline | `v49` + $+200$ s convective thunderstorm offset |
| **Candidate 1: Calibrated Frost** | 375.6755 s | $-0.0474$ s | Airport-specific de-icing offsets |
| **Candidate 2: Fog / LVP Hold** | 375.7065 s | $-0.0164$ s | $+30$ s on non-frost fog flights |
| **Candidate 3: Freezing Precip (`FZ`)** | 375.6901 s | $-0.0328$ s | $+100$ s boost on freezing precip |
| **Candidate 4: July Rain** | 375.7210 s | $-0.0019$ s | $+40$ s on uncovered July rain |
| **Candidate 5 (v51: Combined Best)** | **375.6320 s** | **$-0.0909$ s** | All calibrated weather mechanisms combined |

Candidate 5 demonstrates an offline reduction of **$-0.0909$ s** relative to `v50` with zero regression across individual months.

---

## 3. Specific Adjustments Applied in `v51`

Applied strictly to flights without ground ADS-B transponder tracking:

1. **Airport Frost Calibration (January):**
   - `LFPG` (Paris): $+15$ s beyond `v49` (total $+45$ s de-icing offset; 2,223 flights)
   - `LEMD` (Madrid): $+30$ s beyond `v49` (total $+60$ s de-icing offset; 41 flights)
   - `EHAM` (Amsterdam): $+50$ s beyond `v49` (total $+80$ s de-icing offset; 311 flights)
   - `EGLL` (London) & `EDDM` (Munich): $-30$ s (reverting `v49` offset to 0 s; 322 flights)
2. **Severe Freezing Precipitation (`FZ` in January):**
   - $+80$ s additional offset (80 flights)
3. **Winter Fog / Low Visibility Procedures (January non-frost flights):**
   - $+25$ s hold offset for flights under `FG`/`BR` or visibility $< 1.0$ mile (2,585 flights)
4. **Summer Rain (July non-TS flights):**
   - $+35$ s hold offset for flights under active rain (207 flights)

**Total 2026 Test Set Flights Modified:** 5,743 flights (1.67% of total test set).  
**Mean Shift on Modified Flights:** $+20.67$ s.  
**RMS Distance from `v50`:** 3.4017 s.

---

## 4. Pre-registered Expectations and Pass/Fail Criteria

- **Prior Benchmark:** `v50` = `271.8748`
- **Expected Board Score:** **`271.72` to `271.82`** (predicted delta: $-0.05$ to $-0.15$ s).
- **PASS Criteria:** Score $< 271.8748$. If it passes, `v51` becomes our new official Best submission.
- **FAIL Criteria:** Score $\ge 271.8748$. If it fails or is neutral, `v50` remains our active Best and Last submission on the competition server.
- **Upload Schedule:** Staged and ready for upload when the daily quota resets at 00:00 UTC (08:00 local time).
