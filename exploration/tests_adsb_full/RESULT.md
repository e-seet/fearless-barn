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

## Round 3 (`PREREGISTERED_v45.md`): Sep–Oct 2025 fresh surface

Board so far: **v43 = 281.8629 and v44 = 279.1517**; both beat even the naive projection.

**Surface:** Sep–Oct 2025, 61 days. The base is the forward-val predictions (models trained on
Jan–Aug only), with 369,624 rows.

**Q1, a bigger training set:**
- base 259.32;
- the saved v44 corrector: 251.02 (−8.30 s, 61/61 days better than base);
- the same corrector refitted on Jan + Jul + Nov + Dec: 250.77, winning 49/61 days against the v44
  corrector, so it is **adopted**.

**Q2, more airports** (all three added):

| airport | Sep–Oct | days better | best permutation control | coverage |
|---|---|---|---|---|
| LSZH | 157.1 → 126.2 | 61/61 | 157.8 | 89 % |
| LEMD | 184.0 → 161.3 | 61/61 | 185.2 | 91 % |
| EGLL | 232.5 → 229.6 | 50/61 | 232.7 | 22 % |

**Combined** (7 airports, trained on Jan + Jul + Nov + Dec): 248.23 on Sep–Oct against the 4-airport
corrector's 250.77, so PASS.

**v45 candidate** (`model_v45/`):
- 187,717 rows differ from v42 (54.7 % corrected);
- it rebuilds byte-identical, and v44 still rebuilds byte-identical;
- naive projection −9.15 s against v42.

**Expectation written before upload: 275.0–277.5.** If it scores above 279.1517, v44 is re-uploaded
last.

**Risks, stated before upload:**
- **Coverage shift at EGLL:** 2025 Jul coverage was 12 % against 97 % in 2026 Jul, and the Sep–Oct
  gain there is small.
- **LEMD:** no Jan 2025 coverage, so it is trained on Jul, Nov and Dec.
- **Different base bias in Nov–Dec:** the Nov–Dec training rows carry a +15 s base bias (a horizon
  effect of the forward model), which is not present in v42.

## Round 4 (`PREREGISTERED_v46.md`): Aug 2025 fresh surface

Board: **v45 = 273.5977** (−5.55 vs v44; the written expectation was 275.0–277.5).

**Surface:** Aug 2025, 31 days. The base is the forward_inner val predictions (models trained on
Jan–Jun only), with 191,182 rows.

**Q1, add Sep–Oct to training:** adopted.
- base 271.97;
- the saved v45 corrector: 263.46 (−8.51 s, 31/31 days better than base);
- trained on Jan + Jul + Sep–Dec: 263.20, winning 29/31 days against the v45 corrector.

**Q2, the capacity variant** (600 trees, 31 leaves): 262.11, winning 31/31 days, so **adopted**.

**Q3, LIRF: rejected.** 654.0 → 674.5 (worse); 12/31 days better; still worse with the 5 best rows
removed.

**v46 candidate** (`model_v46/`):
- same 7 airports and the same corrected rows as v45;
- 187,750 rows differ from v42;
- it rebuilds byte-identical, and v45 still rebuilds byte-identical;
- **expectation written before upload: 271.8–273.2.** If it scores above 273.5977, v45 is
  re-uploaded last.

## Round 5 (`PREREGISTERED_interactions.md`): ADS-B × tabular interactions (the cheap stand-in for "ADS-B inside the tree")

Board: **v46 = 272.2619** (−1.34 vs v45; inside its written expectation of 271.8–273.2).

**Design:** leave one month out over Jan, Jul, Aug, Sep, Oct, Nov and Dec 2025, each with its
held-out base.
- **A** = the v46 corrector inputs.
- **B** = A plus all 56 `with_aobt` tabular columns.
- **B_perm** = B with only the ADS-B columns shuffled.

| month | base | A | B | B_perm |
|---|---:|---:|---:|---:|
| Jan | 358.72 | 349.90 | 349.87 | 359.61 |
| Jul | 389.26 | 382.19 | 381.44 | 389.93 |
| Aug | 271.97 | 262.12 | 261.50 | 272.15 |
| Sep | 285.96 | 274.66 | 273.29 | 286.02 |
| Oct | 229.90 | 214.71 | 213.17 | 229.75 |
| Nov | 286.87 | 271.06 | 269.94 | 285.88 |
| Dec | 262.33 | 242.54 | 241.19 | 260.67 |
| **pooled** | 301.94 | 290.29 | **289.39** | 301.91 |

**PASS:** B is better in 7/7 months and −0.90 pooled. B_perm sits at the base level, so the gain
comes from ADS-B interactions, not from refitting the tabular columns.

**v47** (`build_v47.py`, `model_v47/`):
- B trained on all 7 months (627,042 covered rows), at the same 7 airports and with v46's parameters;
- 186,726 rows differ from v46; mean |v47 − v46| on corrected rows is 41 s;
- it reloads from the saved model byte-identical;
- the ~1.5 % of rows with no flight record are missing in both years and are treated identically.

**Expectation written before upload: 271.0–272.0.** If it scores above 272.2619, v46 is re-uploaded
last.
