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

## Reproducibility and a bias check

- **Saved model:** the corrector is saved in `model/adsb_corrector.txt`, with its config in
  `model/config.json`.
- **Rebuild:** `predict_adsb.py --check v43_candidate.parquet` rebuilds the candidate from the saved
  model, v42 and the ADS-B day files. All 344,841 rows and the file bytes are identical.
- **Bias check:** on covered 2025 rows at the three airports, the held-out base underpredicts by
  +6.5 s on average (Jan +7.1, Jul +6.1). The 2026 correction on v42 averages +18 s.
  - Some of that shift may be feature-driven and some a mismatch with v42.
  - If the extra ~12 s were entirely misplaced, it would cost about 0.1 s of RMSE, which is small
    against the expected gain.

## Extension (`PREREGISTERED_extension.md`): a fresh season (Nov–Dec 2025) and more airports

Nov–Dec 2025 is the forward-split test: all 61 days, with a base whose models were trained on
Jan–Aug only.

**A. The frozen v43 corrector on Nov–Dec: PASS.**
- Pooled 274.75 → 267.51 (−7.24 s); 61/61 days better; −7.22 s with the best row removed.
- Per airport: EDDF 185.6 → 142.7, EHAM 189.3 → 146.0, LEBL 178.7 → 153.7.

**B. Per-airport extension** (all three conditions required):

| airport | Jan → Jul | Jul → Jan | Nov–Dec (days better) | verdict |
|---|---|---|---|---|
| EDDM | 181.9 → 164.4 | 177.1 → 165.6 | 215.1 → 131.9 (61/61) | **ADD** |
| LSZH | 199.3 → 171.4 | 204.5 → **206.0** | 163.0 → 131.8 (56/61) | no (fails Jul → Jan) |
| EGLL | 297.9 → 296.0 | 238.2 → **263.4** | 218.3 → 205.9 (52/61) | no |
| LEMD | 189.7 → **192.2** | no Jan 2025 coverage (n = 2) | 170.1 → 149.9 (61/61) | no |

**v44 check:** a single corrector refitted on EDDF, EHAM, LEBL and EDDM scores 263.41 on Nov–Dec,
against the frozen v43 corrector's 267.51, so PASS.

**v44 candidate** (`final_build.py EDDF,EHAM,LEBL,EDDM v44`, saved in `model_v44/`):
- 104,703 rows differ from v42 (30.6 % corrected);
- it rebuilds byte-identical;
- **expectation written before upload: −1.8 to −2.7 s against v42, a board score of 283.8–284.7**
  (naive −5.40 s).

**Not added under this bar:** LSZH and LEMD improve strongly on Nov–Dec but fail one Jan ↔ Jul
direction. Adding them now would be chosen after seeing the result, so it would need a new bar on
data not yet seen. None is left for Jan + Jul 2025.
