# ADS-B corrector at LIRF (flights with an NM record) — result: PASS, v55 built

Pre-registration: `PREREGISTERED.md`. The base is the shipped LIRF recipe (0.541 MLP + 0.459 TabNet,
held out).

**Leave one month out** over Jul–Dec 2025 (LIRF RMSE):

| month | base | corrected | change | eligible rows |
|---|---:|---:|---:|---:|
| Jul | 863.0 | 853.4 | −9.6 | 13,244 |
| Aug | 602.9 | 595.1 | −7.8 | 12,587 |
| Sep | 582.3 | 582.0 | −0.3 | 2,358 |
| Oct | 453.6 | 452.2 | −1.4 | 5,173 |
| Nov | 569.9 | 569.3 | −0.6 | 4,371 |
| Dec | 417.6 | 418.0 | +0.4 | 7,326 |
| **pooled** | **607.77** | **603.58** | **−4.19** | |

**PASS:**
- better in 5/6 months;
- with the best 5 rows removed: 607.13 → 603.16;
- the best permutation control is 608.57, worse than the base.

**Why it works now when round 4 failed:** both training and correction are limited to departures with
a flight record and a track. That leaves out the no-record artifact rows, which ADS-B cannot explain.

**v55** = v50 plus this corrector:
- trained on all 6 months (45,059 rows);
- applied to January 2026 LIRF departures with a record and a track: 6,960 of 11,061 (63 %);
- 6,913 rows change, all of them Jan LIRF;
- correction mean −5.8 s, mean |.| 62.5 s;
- byte-identical from the saved boosters.

**Expectation written before upload: 271.52–271.90, central about 271.80.**
- The gain is concentrated in Jul–Aug.
- In Sep–Dec, the months nearest to January, it is 4–5× smaller per corrected row.
- January itself cannot be validated, because Jan 2025 has no LIRF ADS-B.
- If it scores above 271.8748, v50 is re-uploaded last.
