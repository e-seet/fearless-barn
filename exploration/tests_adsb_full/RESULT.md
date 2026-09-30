# ADS-B (adsb.lol) full test — result

Pre-registration: `PREREGISTERED.md` (main bar, addenda 1–4). Everything below was run under a bar that
was written before the run.

## Data

- 124 days: all of Jan + Jul 2025 and all of Jan + Jul 2026.
- The first fetcher silently truncated 10 days. They were found, deleted and fetched again with a
  fixed fetcher that resumes dropped connections and refuses to write an incomplete day.

## Verdicts, in order

| step | result |
|---|---|
| 18-day test, variants `main` / `move` / `all` | PASS (−1.09 / −0.72 / −1.29 s pooled, mean of both directions) |
| Drift gate (2025 → 2026, same month) | `n_ground`, `path_km` and `n_stops` drift, so they are dropped |
| Drift-clean re-run | `all_nd` PASS (focused −1.83 s); `main_nd` focused FAIL; `move_nd` FAIL |
| Shipped configuration: `all_nd` at EDDF / EHAM / LEBL (18 days) | Jan → Jul −2.71, Jul → Jan −4.89 s; 18/18 days better |
| **Confirmation on 44 fresh days** (addendum 4, frozen configuration) | **PASS:** Jan → Jul 317.20 → 312.56 (−4.63), Jul → Jan 384.21 → 381.71 (−2.50); 44/44 days better; the same with the best row removed; permutation controls +0.2 |
| Drift on the complete 2026 set | the shipped features have \|SMD\| ≤ 0.13; the same three features drift as before |

## What the gain is

- **Where:** at the three airports only, and only on flights with a strictly matched ADS-B ground
  track. Per-airport RMSE drops of 9–29 s on the held-out months.
- **Why the pooled gain is small:** it is diluted by the 7 other airports and by the LIRF tail.
- **What the features are:** post-pushback taxi information:
  - first sighting, first displacement and first movement;
  - time stopped;
  - time from the last stop to liftoff;
  - taxi speed.
- **What it does not do:** it does not recover the recorded off-block time. The first ground sighting
  lags it by 4–11 min at most airports.

## Candidate (`final_build.py`, `build_report.txt`)

- **Training:** all 62 Jan + Jul 2025 days.
- **Base:** applied on top of v42.
- **Output:** `v43_candidate.parquet`, 344,841 rows, same format as v42; 79,761 rows differ from v42
  (23 %).
- **Coverage in 2026:** EDDF 45 %, EHAM 94 %, LEBL 93 %.
- **Correction size:** mean +18 s, mean absolute 67 s, 1–99 % [−172, +427].
- **Written expectation (before any upload):** naive projection −3.97 s against v42's 286.5256;
  **expected −1.3 to −2.0 s**. If it scores above v42, v42 is re-uploaded last as a byte copy.

## Not uploaded: the rules point

The forum question to the moderator mentions "the earlier request to treat post-pushback ADS-B as
leakage". The only reply ("open data sources can be usable") does not address that. These features
are post-pushback, so the upload waits for the user's decision after checking with the moderator.
