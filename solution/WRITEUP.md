# fearless-barn — PRC Data Challenge 2026: taxi-out time

Team **fearless-barn**. This document describes the submitted model, how each design decision was
made, the results, and how to reproduce them.

| | |
|---|---|
| **Final round file** | `submissions/fearless-barn_final.parquet`: 670,790 departures (Jan, Feb, Jun, Jul 2026) |
| **Public leaderboard score** | **269.9888** RMSE (v61). The leaderboard scores January + July 2026 only (344,841 departures, all airports pooled) |
| **Final round score** | none yet: the final ranking is blind |
| **Same base model without ADS-B** | 286.5256 (v42, the competition's data only) |
| **One command** | `python final_pipeline.py --movements <file> --fetch --out <pred>` (§8) |
| **Licence** | GNU GPLv3 (`LICENSE` at the repository root) |

**Contents**

1. [Task and what makes it hard](#1-task-and-what-makes-it-hard)
2. [The model](#2-the-model)
3. [How decisions were made](#3-how-decisions-were-made)
4. [Use of the public leaderboard](#4-use-of-the-public-leaderboard)
5. [The final round](#5-the-final-round)
6. [Results](#6-results)
7. [What did not work](#7-what-did-not-work)
8. [Reproducibility](#8-reproducibility)
9. [Limitations and the ADS-B trade-off](#9-limitations-and-the-ads-b-trade-off)
10. [Data sources and licences](#10-data-sources-and-licences)
11. [Other teams' work: credits and comparison](#11-other-teams-work-credits-and-comparison)
12. [Repository map](#12-repository-map)

---

## 1. Task and what makes it hard

- **Task:** predict the taxi-out time (seconds from off-block to takeoff) of every departure at ten
  European airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM.
- **Training data:** all of 2025, 2,084,659 labelled departures with their movement and flight
  records.
- **Scored sets:**
  - the public board: January + July 2026, 344,841 departures;
  - the final round (added 2026-10-08, blind): January, February, June and July 2026, 670,790
    departures.
- **Metric:** pooled RMSE.

**The metric shapes the whole design.** About 0.02 % of departures (taxi-outs over 2 hours) carry
15–56 % of the squared error, depending on the period. Most of them are record artifacts at Rome
(LIRF): a missing Network Manager (NM) flight record, and an off-block time copied from the schedule.
Under squared error the best prediction is the conditional **mean**, so every method that shrinks the
tail (winsorizing, robust losses, prediction caps) loses (§7).

## 2. The model

### 2.1 Overview

Six stages. Stage 1 predicts every departure from the competition's data. Stages 2–6 correct that
prediction where extra evidence exists.

```
 movements file ──► 1. BASE MODEL   (the competition's data only)
                       per-airport blend of trees + MLPs + FT-Transformers + CatBoost; TabNet at LIRF
                    │
 adsb.lol archive ──► 2. ADS-B CORRECTOR   (EDDF EHAM LEBL EDDM LSZH LEMD EGLL; departures with a track)
                       LightGBM on the base's residual, from the departure's own ground track
                    │
 METAR ─────────────► 3. WEATHER OFFSETS   (departures without an ADS-B correction)
                       frost in January +30 s; thunderstorm in July +200 s
                    │
 adsb.lol archive ──► 4. LIRF CORRECTOR    (departures with an NM record and a track)
                    │
 adsb.lol archive ──► 5. LFPG CORRECTOR    (departures with an NM record and a track)
                    │
 flight plan (LOBT) ► 6. LOBT WINDOW       (clip to ± 3,606 s around takeoff − LOBT)
                    ▼
                    prediction
```

**Two designs are named in this document:**
- **v61** is the best public board file. It has stages 1–5, with stage 4 in January only.
- **The final design** is what produced the final round file: v61's stages, with stage 4 in every
  month and stage 6 added. Both changes were tested for the final round (§5).

`final_pipeline.py` runs the final design by default; `--design v61` reproduces v61.

### 2.2 Stage 1: the base model

**Features** (57, the `with_aobt` featureset, built identically for training and scoring):
- **Calendar and time of day:** hour, day of week, day of year, cyclic encodings.
- **Timing from the flight plan and the NM flight record,** relative to takeoff: scheduled, EOBT,
  IOBT, LOBT and NM actual off-block (AOBT), their differences, and flags for missing records.
- **Categoricals:** airport, runway, stand, destination, aircraft type, operator, wake category,
  market segment, flight type and rules, plus the interactions stand × runway, airport × stand and
  airport × runway.
- **Traffic from the movement records:** departures and arrivals in 15-, 30- and 60-min windows,
  same-runway departures, recent taxi-in times, aircraft taxiing at pushback, and a surge ratio.

Several features use information the competition supplies for every scored row but that a forecast
made at off-block would not have: all timings are measured **from the recorded takeoff**, the NM
**actual** off-block is an input, and some traffic counts include departures **after** the flight's
takeoff (§9).

**Members and blend:**

```
                    ┌─ trees:     per-airport + per-operator LightGBM, one global blend weight
per-airport blend ──┼─ MLP:       entity-embedding MLP (256-128-64), mean of 18 seeds
 (4 weights per     ├─ FT-Transformers, one quarter each of four shapes:
  airport)          │     small d64×3 blocks (6 seeds), large d192×3 (9), deep d96×6 (6), wide d256×2 (6)
                    └─ CatBoost:  mean of 6 seeds
        └─ at LIRF only:  0.541 × MLP + 0.459 × TabNet (mean of 5 seeds)
```

- **Per-airport blend weights** are solved on `forward_inner` (train Jan–Jun 2025, held out
  Jul–Aug), a surface no scored model is fitted on, then frozen
  (`tests_ftt_wide/final_weights_v59_s4.json`). CatBoost takes 0.30–0.55 at LFPG, EGLL, EDDF, LEBL
  and LTFM; the FT-Transformers carry most of EHAM, LSZH and LTFM. At LIRF the solver puts all
  weight on the MLP.
- **Target and clipping:** every valid row is kept, tail included; predictions are clipped to
  [60, 86,400] s.
- **Final refit:** all members are refit on the full year, with hyperparameters and epoch counts
  frozen from time-ordered splits.

**Two versions of this base appear in the results:**
- **v42:** three families (trees, MLP, three FT-Transformer shapes). Board 286.53. Versions v43–v57
  are built on it.
- **v59:** v42 plus the wide FT-Transformer shape and CatBoost as a fourth member
  (`tests_ftt_wide/PROBE.md`). It is the base of v58–v61 and of the final round file.

### 2.3 Stage 2: the ADS-B corrector

**Data.** adsb.lol `globe_history` daily archives (open data, ODbL): all of 2025, and January,
February, June and July 2026. About 1.2 TB were streamed for the first ranking set alone. Each
archive is read once, and only the matched points near the airports are kept
(`tests_adsb_full/fetch.py`). The fetcher resumes dropped connections and refuses to write a
truncated day.

**Matching** (`tests_adsb_full/features.py`):
1. **Candidates:** an aircraft that carried the departure's callsign (`FLIGHT_mvt` or
   `CALLSIGN_flt`) within 10 km of the airport, from 4 h before to 10 min after takeoff.
2. **Strict match:** the aircraft is kept only if its ground → air transition falls within ±120 s
   of the recorded takeoff. This rejects reused callsigns, inbound legs and wrong aircraft.
3. **Coverage:** 44–62 % of departures in 2025–26. LFPG is about 5 % and LTFM 0 %. LIRF is 63 % in
   January 2026 and almost none in July 2026.

**The 7 features:**
- seconds before takeoff of the first ground sighting, the first displacement from the stand and
  the first movement;
- the liftoff offset;
- total stopped time;
- time from the last stop to liftoff;
- median taxi speed.

Three more candidates (sightings count, path length, number of stops) shift between 2025 and 2026,
because the receiver network changed, so they are excluded. Features that describe *how* the aircraft
moved carry over between years; features that describe *how well it was heard* do not.

**Model.** LightGBM on the residual `y − base`, with the 7 features, the base prediction, the
airport and the hour as inputs.
- **Training rows:** departures with a track at the seven airports, January and July–December 2025.
- **Training target:** the residual of **held-out** base predictions (from models that never saw
  the month in question), weighted per airport like the shipped blend.
- **Size:** 3,000 trees, 127 leaves, learning rate 0.015; seeds 42–46 averaged
  (`tests_adsb_full/model_v61`).
- **Rows corrected:** 188,635 of the first ranking set (54.7 %); 359,979 of the final set.

**What it can and cannot see.** Receivers rarely see aircraft at the stand, so the first sighting
lags the recorded off-block by 4–11 min. ADS-B therefore informs the **taxi** (stops, the wait for
the runway, speed), not the off-block record.

### 2.4 Stage 3: weather offsets

The base model has no weather inputs, and the ADS-B corrector sees weather delays only through the
aircraft's own track. Departures **without** an ADS-B correction therefore get two fixed offsets,
from the last METAR within 90 min before takeoff:
- **January frost** (temperature ≤ 34 °F and temperature − dew point ≤ 3 °F): **+30 s**;
- **July thunderstorm** (`TS` in the weather codes): **+200 s**.

**How much to trust them:**
- **Chosen in-sample.** Both values were hand-chosen on January + July 2025.
- **Mildly supported out of sample.** Fitted on 6 months and scored on the 7th, they give −0.07
  pooled (`tests_weather_validated/`).
- **Scoped to their months.** Extending them to every month was tested for the final round and
  rejected (§5), so February and June get no weather offset.

### 2.5 Stages 4 and 5: single-airport ADS-B correctors

| stage | airport | rows | trained on | model |
|---|---|---|---|---|
| 4 | LIRF | departures with an NM record and a track | Jul–Dec 2025 | `tests_lirf_adsb/model_v55` |
| 5 | LFPG | departures with an NM record and a track | Jul–Dec 2025 | `tests_round9/model_v56` |

Both are LightGBM residual models (600 trees, 31 leaves, 5 seeds) on the same 7 features.

- **Why only rows with an NM record.** A corrector trained on all LIRF rows failed (August 2025:
  654 → 675), because the no-record rows are record artifacts that ADS-B cannot explain. Restricted
  to rows with a record it passed (−4.2 RMSE at LIRF, 5 of 6 months).
- **Stage 4's months.** Up to v61 it ran in January only, because LIRF had almost no tracks in July
  2026. In the final design it runs in every month (§5).

### 2.6 Stage 6: the LOBT window

In 2025, every departure with a LOBT has |off-block − LOBT| ≤ 3,606 s. So the taxi-out time must lie
within 3,606 s of (takeoff − LOBT). Predictions are clipped to that window. The idea is from team
outgoing-boat (§11); it was re-derived on our data and tested for the final round (§5). It moves 65
of the 670,790 final rows.

### 2.7 What each stage is worth

Public board (January + July 2026), in the order the stages were added:

| step | board | change |
|---|---:|---:|
| stage 1, v42 base | 286.53 | |
| + stage 2 (7 airports; 600 trees) | 272.17 | −14.36 |
| + stage 3 (weather offsets) | 271.87 | −0.30 |
| + stage 4 (LIRF corrector) | 271.76 | −0.11 |
| + stage 5 (LFPG corrector) | 271.56 | −0.20 |
| stage 2 at 1,500 trees | 270.88 | −0.68 |
| stage 1 → v59 base (wide FT-Transformer, CatBoost) | 270.81 | −0.07 |
| stage 2 at 3,000 trees | 270.47 | −0.34 |
| stage 2 trained on a per-airport weighted base (**v61**) | **269.99** | −0.48 |

Stage 6 and the every-month stage 4 have no board score: they were added for the blind final round.

## 3. How decisions were made

**Splits.** All splits are time-ordered; nothing is selected on the full-year refit.

| split | train | held out | used for |
|---|---|---|---|
| `forward` | Jan–Aug | Sep–Oct (val), Nov–Dec (test) | early stopping, comparison |
| `forward_inner` | Jan–Jun | Jul–Aug | blend weights, solved out of sample |
| `seasonal_full` | Feb–Jun, Aug–Nov | Dec (val), **Jan + Jul** (test) | the board's months |
| `holdout_febjun` | Jan, Jul–Nov | Dec (val), **Feb–Jun** (test) | the final round's new months |
| `full` | all of 2025 | — | the final refit only |

**Rules applied to every experiment:**
- **The pass bar is written before the result exists.** Every test has a `PREREGISTERED*.md`,
  committed before it ran, with its decision rule and the action for pass or fail.
- **Confirmation on data the choice never saw.** Each ADS-B round was confirmed on a fresh 2025
  season; later rounds used leave-one-month-out over seven months of held-out base predictions.
- **Controls:**
  - permutation controls (the new inputs shuffled) must lose;
  - the result must survive removing the largest-gain rows (5, later 20), because the tail can fake
    a gain;
  - several seeds, paired by seed.
- **Drift gate for any external input:** 2025 against 2026 in the same months; a standardized
  shift above 0.2 disqualifies the feature.
- **A training run that hit its epoch cap is unresolved, not failed.** A convergence audit re-read
  every early verdict.

**Two lessons that changed the rules:**
- **A within-2025 holdout cannot see a 2025 → 2026 shift.** v47 (tabular columns inside the ADS-B
  corrector) won 7/7 months locally and lost 0.47 on the board. The drift gate exists because of it.
- **The reference must carry every fixed rule the candidate carries.** The copy-mixture test (v63)
  passed its bar (−1.01 pooled, 6/7 months), but the candidate carried the LOBT clip and the
  reference did not. Against a clipped reference it was +0.22 worse on January + July, and the
  board gave +0.21. Since then, gains are judged on the board's months and on the body of the
  distribution, not on the tail.

## 4. Use of the public leaderboard

The organizers asked teams not to optimize for the ranking. This section states exactly what we did.

- **45 distinct files were uploaded** (from 2026-09-08). Each was a full 344,841-row prediction
  file, with its expected score and fallback written down beforehand
  (`submissions/board_scores.csv`). v61 was uploaded a second time on 2026-10-08, when we still
  believed the ranking used the last upload.
- **No upload altered single rows to read their labels.**
- **Four uploads add a correction to a defined subset** of departures: v49 and v50 (weather
  conditions), v55 (LIRF tracked flights), v56 (LFPG tracked flights). Each was validated locally
  first.
- **What a subset correction reveals, and why it was not used.** Adding a constant to a fixed subset
  and reading the score change reveals that subset's mean error on the hidden labels. v49's result
  implies the January frost subset is under-predicted by about 160 s on average, against the +30 s
  applied. That number was deliberately **not** used to set any later value.
- **One constant was fitted on the board:** the LIRF TabNet weight, 0.459. LIRF's prediction is
  linear in that weight, so the board's squared error is a parabola through three existing uploads.
- **The weather offsets were hand-chosen** on January + July 2025 (§2.4).
- **Four weather-offset variants (v51–v54) were built and not uploaded,** because uploading several
  to keep the best would have tuned to the hidden labels. They sit in `submissions/DO_NOT_UPLOAD/`.
- **A 45-row file (v62, the LOBT clip alone) was not uploaded,** because a 45-row change is too
  close to probing.
- **Where the board was the only usable measurement.** Local tests under-read averaging and model
  variety (MLP 1 → 3 seeds: −0.9 locally, −18 on the board). For those questions a single upload
  with a written fallback was used.
- **The final round file was built without any board feedback** (§5).

## 5. The final round

On 2026-10-08 the organizers added a blind final ranking: one file for `final_submitting.parquet`
(670,790 departures of January, February, June and July 2026), scored on the first two months, the
two new months and all four.

### 5.1 Re-checking the design on the new months

Several stages were scoped to the first ranking set's months. Each was re-checked on **held-out
February–June 2025** with the saved correctors, none of which was trained on those months. The bar
was pre-registered (`tests_final/PREREGISTERED.md`): better on Feb + Jun, better on Feb–Jun, better
in at least 2/3 of the months it changes, and still better with the best 20 rows removed.

**The design transfers to the new months:**

| | Feb–Jun 2025 | Feb + Jun 2025 |
|---|---:|---:|
| stage 1 | 285.11 | 327.74 |
| + stage 2 and stage 5 | 274.13 | 317.39 |

Stage 2 is worth −11.0 there (5/5 months; −11.4 with the best 20 rows removed).

**Each proposed change, against that design:**

| change | rows | Feb–Jun | Feb + Jun | months | best 20 removed | decision |
|---|---:|---:|---:|---:|---:|---|
| stage 4 (LIRF corrector) in every month | 13,009 | −0.30 | −0.47 | 2/2 | −0.19 | **adopted** |
| stage 6 (LOBT window) | 86 | −0.11 | −0.11 | 5/5 | −0.005 | **adopted** |
| frost +30 s in every month | 3,305 | −0.14 | −0.31 | 1/3 | −0.12 | rejected |
| thunderstorm +200 s in every month | 827 | −0.06 | −0.15 | 3/5 | −0.005 | rejected |
| LIRF no-record specialist | 470 | −15.8 | −32.0 | 4/5 | +3.29 | rejected |
| the same, only > 6 h behind schedule | 14 | −14.3 | −24.5 | 4/5 | +0.009 | rejected |
| context corrector on every row | all | −4.25 | −7.39 | 5/5 | −3.71 | rejected (fails on Jan + Jul) |

**Two rejections were judgement calls, disclosed here:**
- **The LIRF no-record specialist** (an idea from teams quirky-honey and zestful-fountain) gave
  large pooled gains on both surfaces: −14.0 on the seven scored months and −15.8 on Feb–Jun. Its
  whole gain sits in about 20–30 LIRF departures a year with no flight record and a takeoff many
  hours behind schedule. It fails the best-20 bar on both surfaces, and that kind of gain has failed
  to transfer before (our LIRF identity override, §7). We kept to the bar.
- **The context corrector** (neighbouring departures' recent taxi time and lateness, on every row;
  `tests_ctx/`) passed on Feb–Jun but was +0.44 worse on January + July 2025, with almost all of its
  gain in the tail. Its Feb–Jun gain is also likely overstated, because the base it was measured
  against there is weaker than the shipped one. The pre-registration required both surfaces.

**Limitation.** At LIRF the Feb–Jun check used a three-member held-out base without the TabNet
override, which has no held-out predictions for those months. The stage 4 decision was therefore made
on a slightly different base than the one it is applied to.

### 5.2 The final file's data differs from the first ranking set

In `final_ranking.parquet`, the January and July rows are not identical to `ranking.parquet`:
- takeoff times differ on 92 % of rows (median 4 s, at most 12 s);
- about 1 % of flight-record fields differ;
- scheduled times are unchanged.

Two consequences:
- **Predictions are made from the final file's own data.** Time-of-day, delay and traffic-window
  features move with the takeoff times, so the final file's January and July predictions differ from
  v61's by 9.9 s RMS.
- **ADS-B times were re-based.** Each stored ADS-B point carries its time relative to the takeoff
  time in the file it was matched against. `tests_final/rebase_points.py` re-bases the January and
  July points on the final file's takeoff times (316,809 movements, by up to 12 s). 2025-12-31 and
  2026-01-31 were re-fetched against the final file, for the early takeoffs on 1 January and
  1 February. Effect: January rows 11.7 s RMS, July 8.1 s, mean 0.0.

### 5.3 The build

1. **Stage 1:** `python predict_new.py --movements ../prc-2026-datasets/final_ranking.parquet
   --ids ../prc-2026-datasets/final_submitting.parquet --weights tests_ftt_wide/final_weights_v59_s4.json
   --out tests_final/final_base.parquet`
2. **ADS-B:** `sh tests_final/run_fetch.sh` (59 day files for February and June 2026), then
   `python tests_final/rebase_points.py`.
3. **Stages 2–6:** `python final_pipeline.py --movements ../prc-2026-datasets/final_ranking.parquet
   --base tests_final/final_base.parquet --points-dir tests_final/points_final
   --out submissions/fearless-barn_final.parquet`

**Result:** 670,790 rows, with the same IDs and order as the template.

| stage | rows changed |
|---|---:|
| 2. ADS-B corrector | 359,979 |
| 3. frost / thunderstorm | 4,883 / 430 |
| 4. LIRF corrector | 12,551 |
| 5. LFPG corrector | 3,663 |
| 6. LOBT window | 65 |

Departures with ADS-B track points: January 81 %, February 84 %, June 77 %, July 83 %.

## 6. Results

**Public board history** (January + July 2026):

| version | change | board | change |
|---|---|---:|---:|
| v2 | pooled LightGBM | 405.75 | |
| v3 | per-airport trees | 386.22 | −19.5 |
| v13 | + 1-seed MLP in a per-airport blend | 324.28 | −61.9 |
| v14 → v25 | MLP seeds 3 → 18 | 306.32 → 295.21 | −29.1 |
| v29 | + TabNet override at LIRF | 293.67 | −1.5 |
| v30 | + FT-Transformer (small) as a third member | 291.34 | −3.9 vs v25 |
| v31 | v30 + the TabNet override at LIRF | 289.78 | −1.6 |
| v32 → v42 | FT-Transformer seeds and three shapes; TabNet weight 0.459 | 287.99 → 286.53 | −3.3 |
| v43 → v48 | ADS-B corrector, 3 → 7 airports | 281.86 → 272.17 | −14.36 |
| v49, v50 | weather offsets | 271.94, 271.87 | −0.30 |
| v55 | LIRF ADS-B corrector | 271.76 | −0.11 |
| v56 | LFPG ADS-B corrector | 271.56 | −0.20 |
| v57 | 7-airport corrector at 1,500 trees | 270.88 | −0.68 |
| v58 | base: + wide FT-Transformer, + CatBoost | 270.8103 | −0.07 |
| v59 | wide FT-Transformer and CatBoost 3 → 6 seeds | 270.81 | tie |
| v60 | 7-airport corrector at 3,000 trees | 270.47 | −0.34 |
| **v61** | **7-airport corrector trained on a per-airport weighted base** | **269.99** | **−0.48** |

**Standing:** #59 of 120 on the public board on 2026-10-08.

**How each ADS-B step was accepted** before its upload:

| version | change | local test | board |
|---|---|---|---:|
| v43 | 3 airports | Jan ↔ Jul plus 44 fresh days: 44/44 days better | 281.86 |
| v44 | + EDDM | Nov–Dec 2025, frozen corrector: 61/61 days | 279.15 |
| v45 | + LSZH, LEMD, EGLL; + Nov–Dec training | Sep–Oct 2025: 61/61 days | 273.60 |
| v46 | + Sep–Oct training | Aug 2025: 31/31 days | 272.26 |
| v48 | 5-seed average; + Aug training | leave one month out, 7/7 months | 272.17 |
| v55 | LIRF corrector | leave one month out, 5/6 months | 271.76 |
| v56 | LFPG corrector | leave one month out, 6/7 months | 271.56 |

**Uploads that lost.** The ranking uses each team's best submission, so a losing upload costs only
a daily slot.

| version | change | result |
|---|---|---:|
| v18 | prediction cap at 2 h | 491.35 |
| v27 | target winsorized at 2 h | 496.09 |
| v15, v17 | LIRF identity override | +0.86 twice |
| v33 | TabNet 11 seeds | +0.91 |
| v37 | season-weighted MLPs | +2.07 |
| v47 | tabular columns inside the ADS-B corrector | +0.47 |
| v63 | copy-mixture member + LOBT clip | +0.21 |

## 7. What did not work

Every row was tested; records are in the named `tests_*/RESULT.md`.

**Target and loss**

| idea | result |
|---|---|
| winsorize / cap / floor the target or the predictions | large losses (board 491–496) |
| log target, Huber, Poisson, quantile ladder, hurdle model | all de-weight the tail and lose |

**Base model**

| idea | result |
|---|---|
| MLP variants, PLE embeddings, TabM, season weighting, transductive pretraining | all worse |
| sequence and neighbour models (GRU, queue attention, retrieval) | residuals 0.98 correlated with the MLP |
| XGBoost, tree-side seeds, CatBoost as a third member (v26) | no gain once blended. CatBoost was re-tested as a fourth member after an audit of early closures and is in the v59 base (§2.2) |
| METAR weather as model features; holidays; OSM stand distance | within noise |
| runway queue / runway-mode features | stable across years, but +2.1 worse as a corrector |
| routed taxi distance (apt.dat), measured as a ceiling | at most −0.14 s; not built |

**ADS-B**

| idea | result |
|---|---|
| ADS-B features inside the tree model | tree −7 alone, +0.08 once corrected |
| traffic counts from other aircraft's ADS-B | fail the drift gate (they track receiver coverage) |
| tabular columns inside the ADS-B corrector (v47) | 7/7 months locally, +0.47 on the board |
| Feb–Jun as extra corrector training | +0.07, 2/7 months |
| looser matching at EDDF / LEMD | the extra matches have no ground points |
| stage 2 and stage 5 trained on the four-member base (round 13) | 4/7 months; fail |

**The tail and the label**

| idea | result |
|---|---|
| LIRF identity override (no record → scheduled delay) | right in 2025, +0.86 on the 2026 board twice |
| copy-mixture member: copy classifier + normal regressor (v63) | Jan + Jul +0.22 against the clipped recipe; board +0.21 |
| per-airport line on delay-to-schedule for no-record rows over 6 h | −12.7 locally, all from about 10 LIRF rows; worse without the best 20 |
| LIRF no-record specialist and its > 6 h form | large pooled gains in a few dozen tail rows; fail the best-20 bar (§5) |
| context corrector on every row | Jan + Jul +0.44; gain almost all tail (§5) |

**One lesson stands out: an idea closed on a weak test may be wrong.** ADS-B was first "closed" on
three days and one crude feature. Tested properly, it became the largest gain since the first neural
model.

## 8. Reproducibility

**One command, for any movements file in `ranking.parquet`'s schema:**

```bash
python final_pipeline.py --movements new.parquet --points-dir adsb_new --fetch --out pred.parquet
```

It does four things:
1. runs stage 1 from the saved models (`predict_new.py`, v59 base);
2. streams the adsb.lol archives for the departure days (`tests_adsb_full/fetch.py`);
3. fetches any METAR reports the local weather file lacks, from the Iowa Environmental Mesonet;
4. applies stages 2–6 with the saved correctors.

**Environment:** Python 3.12, `requirements.txt`. Training used one NVIDIA RTX 3080 Ti; prediction
runs on CPU, with a GPU used for the FT-Transformers when present.

**Verified:**

| check | command or evidence | result |
|---|---|---|
| v61 from the saved v59 base | `python final_pipeline.py --design v61 --base tests_ftt_wide/v59_base.parquet --check submissions/fearless_barn_v61.parquet --out v61.parquet` | byte-identical, 344,841 rows |
| v56 from the saved v42 base | the same with `--base submissions/fearless_barn_v42.parquet --stage2-model tests_adsb_full/model_v48 --check submissions/fearless_barn_v56.parquet` | byte-identical |
| stage 1 from raw data (v59) | `python predict_new.py --weights tests_ftt_wide/final_weights_v59_s4.json --check tests_ftt_wide/v59_base.parquet` | 344,840 / 344,841 rows identical; one LIRF row off by 1 s (a TabNet rounding boundary) |
| new data end to end | three 2025 days treated as new data (15,408 departures), fresh ADS-B downloads | identical to the run on the existing ADS-B files |
| every ADS-B candidate, v43–v61 | rebuilt from its saved models | byte-identical |

**What is in git and what is not:**

| item | in git | notes |
|---|---|---|
| code, pre-registrations, results | yes | |
| stage 2, 4 and 5 corrector models | yes | `tests_adsb_full/model_v*`, `tests_lirf_adsb/model_v55`, `tests_round9/model_v56` |
| blend weights | yes | `tests_ftt_wide/final_weights_v59_s4.json` |
| key prediction files and stage-1 outputs | yes | `submissions/` (v42, v56, v61, v63 and the final file), `tests_ftt_wide/v59_base.parquet`, `tests_final/final_base.parquet` |
| trained stage-1 models and checkpoints (about 37 GB) | no | too large; kept by the team and available on request. Listed with sha256 in `tests_repro/dependency_manifest*.csv` |
| ADS-B day files (about 3.3 GB) | no | re-fetchable with `fetch.py` |
| competition data, METAR files | no | not redistributed; see §10 |

**Without the 37 GB of trained models, stage 1 cannot be re-run.** The saved stage-1 outputs let
stages 2–6 be reproduced from the repository plus the ADS-B and METAR downloads.

## 9. Limitations and the ADS-B trade-off

**Post-pushback ADS-B.** Stages 2, 4 and 5 use each departure's **own** ground track after pushback:
when it moved, how long it stopped, how fast it taxied.
- **What it gives:** 286.53 → 269.99 on the public board.
- **What it costs:** it is not available before off-block. It relies on volunteer receiver coverage,
  which varies by airport and changed between 2025 and 2026.
- **What is new and what is not.** The competition's own inputs are already partly after the event:
  the recorded takeoff time, the NM actual off-block, and later departures' times (§2.2). ADS-B adds
  the aircraft's **own observed taxi**, the closest any input comes to the quantity being predicted.
- **Status.** adsb.lol is open data under ODbL, and the organizers wrote that open data sources may
  be used. We declare its use here.
- **The model without ADS-B is stage 1 alone:** 286.5256 on the public board (v42). For the final
  round it is `tests_final/final_base.parquet`, the output of step 1 in §5.3.

**Other limitations:**
- **Not a pre-departure forecast.** The model reconstructs taxi-out time from records that include
  the takeoff time. It is not a prediction made at the gate.
- **The tail is not solved.** A few dozen multi-hour rows a year, mostly LIRF departures without a
  flight record, carry a large share of the squared error. Every tail-specific model we tested
  failed our robustness bar (§5, §7), so those rows get the base prediction.
- **Coverage.** The ADS-B corrector reaches about 55 % of departures; LTFM has no coverage and LFPG
  very little.
- **Fixed, month-scoped rules.** The weather offsets are two hand-chosen constants for January and
  July.
- **One board-fitted constant:** the LIRF TabNet weight (§4).

## 10. Data sources and licences

| data | use | source / licence |
|---|---|---|
| 2025 training set, 2026 ranking set, final ranking set | everything | PRC Data Challenge 2026 (EUROCONTROL / airports); not redistributed here |
| ADS-B ground tracks | stages 2, 4, 5 | **adsb.lol** `globe_history`, © adsb.lol contributors, **ODbL** |
| METAR, 10 airports | stage 3 | Iowa Environmental Mesonet ASOS archive (public), fetched into `prc-2026-datasets-wx/` |

**Not used:** the OpenSky historical database; OPDI (confirmed as open data by the organizers on
2026-10-08, but its ground coverage is weakest where our ADS-B is also missing).

## 11. Other teams' work: credits and comparison

We read several published repositories for ideas, after our model was built. Four of them gave ideas we tested. None of their code is
used; every idea below was re-implemented and tested by us.

| idea | from | outcome here |
|---|---|---|
| LOBT window | outgoing-boat (github.com/EnioAguiar/prc-taxiout-2026) | **in the final design** (stage 6) |
| copy classifier with an expectation mixture | outgoing-boat | tested as v63, lost |
| line rule for no-record rows over 6 h | outgoing-boat | tested, rejected |
| LIRF no-record specialist | quirky-honey (github.com/dhsdb010/OPDI), zestful-fountain (github.com/Phoenix-Ops-LTD/prc2026-taxiout) | tested, rejected (§5) |
| neighbour taxi level and lateness in a corrector | quirky-honey, outgoing-boat, joris-decombe (github.com/joris-decombe/taxiout) | tested, rejected (§5) |

**Why outgoing-boat (243.95) is ahead of us (269.99), by each team's board history:**

| | outgoing-boat | fearless-barn |
|---|---:|---:|
| model without ADS-B | 331 | **286.5** |
| + ADS-B, cross-fitted corrector, LOBT window | 276 | 272.2 |
| + context and external data in the corrector | 248 | 271.6 |
| + more correctors, rebuilt base | 244 | 270.0 |

Our base model is about 45 s better than theirs. They lead because of what their corrector sees and
which rows it reaches:
- **Coverage.** Their corrector runs on every row at all ten airports, with ADS-B fields as inputs
  that may be empty. Ours runs on about 55 % of rows.
- **Inputs.** Their corrector reads the airport's state: arrivals' taxi-in, runway configuration,
  gate retention, EUROCONTROL daily series, stand-to-runway distances. Ours reads the departure's own
  track and the base prediction.
- **Design around the label.** Their base separates off-block times copied from the schedule (about
  9 % of departures) from real taxi times. Ours learns both in one model.

Our one attempt at an every-row corrector (§5) did not pass. With more time we would rebuild that
stage with their wider inputs, under our stronger base.

## 12. Repository map

Paths are relative to `solution/`. The folder is flat, because the pipeline reads several of these
paths directly. [`README.md`](README.md) in this folder is the full index: every `pipeline_*/` and
`tests_*/` folder, grouped by topic.

**The solution**

| path | contents |
|---|---|
| `final_pipeline.py` | the whole recipe, stages 1–6 |
| `predict_new.py` | stage 1 from saved models, for any movements file |
| `common/`, `config.py` | data loading, features, splits |
| `submissions/` | the final round file, key board files, `board_scores.csv` (every score with its pre-written expectation) |
| `requirements.txt` | Python dependencies |

**Where each stage lives**

| stage | training code and tests | saved model or weights |
|---|---|---|
| 1. base model | `pipeline_airport_operator_blend/`, `pipeline_mlp_antigravity/`, `pipeline_ftt_multimodel/`, `pipeline_catboost_antigravity/`, `pipeline_tabnet_lirf_override/`; `tests_ftt*/`, `tests_tabnet/` | `tests_ftt_wide/final_weights_v59_s4.json`; trained models not in git (§8) |
| 2. ADS-B corrector | `tests_adsb_full/`, `tests_round10/` – `tests_round13/` | `tests_adsb_full/model_v61/` |
| 3. weather offsets | `tests_weather_metar/`, `tests_weather_validated/` | two constants in `final_pipeline.py` |
| 4. LIRF corrector | `tests_lirf_adsb/` | `tests_lirf_adsb/model_v55/` |
| 5. LFPG corrector | `tests_round9/` | `tests_round9/model_v56/` |
| 6. LOBT window | `tests_final/` | a rule in `final_pipeline.py` |

**Everything else**

| path | contents |
|---|---|
| `tests_final/` | the final round: Feb–Jun re-check, ADS-B fetch and re-basing, the build |
| `tests_copymix/`, `tests_lirf_spec/`, `tests_ctx/` | ideas from other teams, tested and rejected |
| other `tests_*/` | every other experiment, each with its pre-registration and result |
| other `pipeline_*/`, `overtuned/` | candidate base models that were not selected |
| `artifacts/` | metadata of every trained run (parameters, metrics, splits) |
| `tests_repro/` | sha256 manifests of the files git does not hold |
| `EXPLORATION_SUMMARY.md` | one-page list of everything tried |
| `plans/`, `PIPELINE_REVIEW.md`, `TAXI_OUT_DRIVERS.md` and the other top-level `.md` files | decision history and early analyses, not maintained |
| `stage*.py`, `scratch_*.py`, `compare*`, `run_*`, `logs/` | early analysis scripts and their outputs, not part of the pipeline |
