# Probe v50: July Convective Thunderstorm Offset for Uncovered Flights — RESULT

**Date:** 2026-10-01  
**Status:** **SUCCEEDED — NEW BEST AND LAST ON LIVE LEADERBOARD**  
**Score:** **271.8748** (improved by **-0.0692 s** over v49's 271.9440, and **-0.2966 s** over v48's 272.1714)  
**Board Standing:** **Rank 46 / 214 teams** (firmly ahead of Rank 47 at 272.3391)

---

## 1. What was built and tested

Following [PREREGISTERED_v50.md](file:///c:/Users/Admin/Desktop/eddie/fearless-barn/exploration/tests_weather_metar/PREREGISTERED_v50.md):
- **Base Submission:** `fearless_barn_v49.parquet` (score 271.9440), which already contains the validated +30s January morning de-icing adjustment.
- **Target Population:** Exactly 428 July 2026 departures meeting two physical criteria:
  1. No ADS-B ground trajectory was available in July 2026 (`v49 == v42` on July rows).
  2. METAR at departure time indicated active thunderstorm (`TS` in `wxcodes`).
- **Target Breakdown by Airport:**
  - LTFM (Istanbul): 370
  - LSZH (Zurich): 21
  - LEMD (Madrid): 18
  - LIRF (Rome): 10
  - EDDM (Munich): 6
  - LFPG (Paris): 2
  - LEBL (Barcelona): 1
- **Modification:** Added $+200.0$ s convective holding offset to those exact 428 rows.
  - At LTFM in 2025, empirical mean taxi time jumped by $+648.9$ s during thunderstorms (from 1051.7 s to 1700.6 s).
  - The $+200$ s adjustment safely brings the average prediction on these flights from 1129.8 s to 1329.8 s, well below the 1700 s empirical mean.
- **Untouched Rows:** Exactly 344,413 rows (99.88% of the dataset) remain 100% byte-identical to `v49`. All January flights are untouched, preserving the full $-0.2274$ s de-icing gain from `v49`.
- **Validation:** Verified with `submit.validate()` (0 missing, 0 extra, 0 negative, 0 non-finite).

---

## 2. Leaderboard Result & Verification

- **Uploaded File:** `fearless-barn_v50.parquet`
- **Result JSON:** `fearless-barn_v50.parquet_result.json`
- **Scored Pairs:** 344,841
- **Live Score:** **271.8748**
- **Board Delta:** $-0.0692$ s vs `v49`, $-0.2966$ s vs `v48`
- **Current Standing:** `fearless-barn_v50.parquet` is now active as both the **BEST** and **LAST** submission on the live leaderboard.
