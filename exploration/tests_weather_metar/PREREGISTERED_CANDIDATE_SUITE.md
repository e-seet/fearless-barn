# Pre-Registration Suite: Candidate Submissions v51, v52, v53, v54

**Date:** 2026-10-01  
**Baseline on Live Board:** `fearless-barn_v50.parquet` — **`271.8748`** (Rank #46 / 214 teams)  
**Parent Model:** `fearless_barn_v50.parquet`  
**All Candidates:** Pre-registered, built, validated through `submit.validate()`, and staged in `exploration/submissions/`.  

---

## 1. Candidate Comparison Matrix

All candidates target the **157,116 departures with zero ADS-B ground transponder coverage** (45.6% of the 344,841 competition test flights, including 100% of Paris LFPG and Istanbul LTFM).

| Candidate | Core Strategic Hypothesis | 2025 Offline Gain vs `v50` | 2026 Flights Modified | RMS Delta from `v50` | Expected Board Score | Risk Profile |
|---|---|---:|---:|---:|---:|---|
| **`v51`** | **Balanced Multi-Component:** Calibrated frost by airport, freezing precipitation boost (+80s), non-frost fog hold (+25s), July rain (+35s) | **$-0.0822$ s** | 5,743 (1.67%) | 3.40 s | `271.72` to `271.82` | **Recommended Primary** — highest all-round validation balance |
| **`v52`** | **Conservative De-icing Specialist:** Pure winter de-icing only. Calibrated frost (LFPG +45s, LEMD +60s, EHAM +80s; EGLL/EDDM 0s) + Jan FZ (+80s). No fog or rain adjustments. | **$-0.0632$ s** | 2,951 (0.86%) | 2.48 s | `271.78` to `271.84` | **Lowest Risk** — isolates pure thermodynamic icing, zero fog/rain noise |
| **`v53`** | **Aggressive Storm & Severe Winter:** LTFM July convective thunderstorms scaled up to $+320$s (matching empirical 2025 $+314$s in-band residual), Jan FZ $+120$s, Jan snow $+40$s, fog $+25$s, rain $+35$s. | **$-0.0845$ s** | 8,047 (2.33%) | 6.19 s | `271.70` to `271.80` | **Highest Upside** — aggressively attacks long taxi delays in convective storms |
| **`v54`** | **LVP & Peak-Hour Congestion Interaction:** Airport capacity drops 50% during Low Visibility Procedures (LVP); morning rush-hour fog (06:00–10:00 UTC) gets $+50$s; off-peak fog gets $+25$s; calibrated frost + rain. | **$-0.0784$ s** | 5,743 (1.67%) | 4.17 s | **Structured Queuing** — explicitly models queue buildup during morning low-visibility operations |

---

## 2. Detailed Adjustments by Component

### A. Airport-Specific Frost Calibration (January, applied in all 4 candidates)
- `LFPG` (Paris): $+15$ s beyond `v49` (total $+45$ s de-icing hold; 2,223 flights)
- `EHAM` (Amsterdam): $+50$ s beyond `v49` (total $+80$ s de-icing hold; 311 flights)
- `LEMD` (Madrid): $+30$ s beyond `v49` (total $+60$ s de-icing hold; 41 flights)
- `EDDM` (Munich) & `EGLL` (Heathrow): $-30$ s (reverting `v49`'s flat offset to $0$ s; 322 flights)

### B. Freezing Precipitation (`FZ`)
- `v51`, `v52`, `v54`: $+80$ s (80 flights in January 2026)
- `v53`: $+120$ s (80 flights in January 2026)

### C. Fog & Low Visibility Procedures (LVP)
- `v51`: Flat $+25$ s on non-frost fog flights (2,585 flights)
- `v52`: Excluded ($0$ s)
- `v53`: Flat $+25$ s (2,585 flights)
- `v54`: Peak morning hours (06:00–10:00 UTC) $+50$ s; off-peak $+25$ s (2,585 flights)

### D. July Convective Storms & Rain
- `v51`, `v52`, `v54`: July TS stays $+200$ s from `v50`; July non-TS rain $+35$ s (207 flights)
- `v53`: July TS at LTFM increased to $+320$ s (370 flights); non-TS rain $+35$ s (207 flights)

---

## 3. Staging and Ready-to-Upload Commands

When the competition quota resets at **00:00 UTC** (08:00 local time), any of these candidates can be uploaded immediately with a single command:

* **To upload `v51` (Recommended Primary):**
  ```bash
  mc cp "c:\Users\Admin\Desktop\eddie\fearless-barn\exploration\submissions\fearless_barn_v51.parquet" dc26/prc-2026-fearless-barn/fearless-barn_v51.parquet
  ```

* **To upload `v52` (Conservative De-icing):**
  ```bash
  mc cp "c:\Users\Admin\Desktop\eddie\fearless-barn\exploration\submissions\fearless_barn_v52.parquet" dc26/prc-2026-fearless-barn/fearless-barn_v52.parquet
  ```

* **To upload `v53` (Aggressive Storms):**
  ```bash
  mc cp "c:\Users\Admin\Desktop\eddie\fearless-barn\exploration\submissions\fearless_barn_v53.parquet" dc26/prc-2026-fearless-barn/fearless-barn_v53.parquet
  ```

* **To upload `v54` (Peak-Hour LVP):**
  ```bash
  mc cp "c:\Users\Admin\Desktop\eddie\fearless-barn\exploration\submissions\fearless_barn_v54.parquet" dc26/prc-2026-fearless-barn/fearless-barn_v54.parquet
  ```

---

## 4. Pass/Fail Decision Criteria
- **PASS:** Board RMSE $< 271.8748$. The candidate becomes our new BEST and LAST submission.
- **FAIL:** Board RMSE $\ge 271.8748$. We immediately revert to `v50` as the active BEST submission.
