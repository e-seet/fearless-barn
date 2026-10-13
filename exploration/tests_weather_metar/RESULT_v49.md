# Probe v49: Winter Morning De-icing Offset for Uncovered Flights — RESULT

**Date:** 2026-10-01  
**Status:** **SUCCEEDED — NEW BEST AND LAST ON LIVE LEADERBOARD**  
**Score:** **271.9440** (improved by **-0.2274 s** over v48's 272.1714; pre-registered expectation was 272.09–272.14)  
**Board Standing:** **Rank 46 / 214 teams**

---

## 1. What was built and tested

Following [PREREGISTERED_v49.md](file:///c:/Users/Admin/Desktop/eddie/fearless-barn/solution/tests_weather_metar/PREREGISTERED_v49.md):
- **Base Submission:** `fearless_barn_v48.parquet` (score 272.1714).
- **Target Population:** Exactly 4,878 January 2026 departures meeting two physical criteria:
  1. No ADS-B ground trajectory was available in `v48` (`v48.TAXITIME_SEC_mvt == v42.TAXITIME_SEC_mvt`, i.e., zero empirical stop/lineup observations).
  2. METAR at departure time indicated frost conditions: ambient temperature $T \le 34^\circ\text{F}$ ($\le 1.1^\circ\text{C}$) and dewpoint depression $T - T_d \le 3.0^\circ\text{F}$ ($\le 1.7^\circ\text{C}$).
- **Target Breakdown by Airport:**
  - LFPG: 2,223
  - EDDF: 1,257
  - LSZH: 377
  - EHAM: 311
  - LTFM: 286
  - EDDM: 242
  - EGLL: 80
  - LIRF: 61
  - LEMD: 41
- **Modification:** Added $+30.0$ s de-icing hold offset to those exact 4,878 rows. All remaining 339,963 rows (all July flights, all ADS-B covered flights, and all non-frost flights) remained 100% byte-identical to `v48`.
- **Validation:** Verified with `submit.validate()` (0 missing, 0 extra, 0 negative, 0 non-finite).

---

## 2. Leaderboard Result & Verification

- **Uploaded File:** `fearless-barn_v49.parquet`
- **Result JSON:** `fearless-barn_v49.parquet_result.json`
- **Scored Pairs:** 344,841
- **Live Score:** **271.9440**
- **Board Delta:** $-0.2274$ s vs `v48`
- **Pre-registered Expectation:** 272.09 to 272.14 (outperformed expectation by ~0.15 s).
- **Current Standing:** `fearless-barn_v49.parquet` is now active as both the **BEST** and **LAST** submission.
