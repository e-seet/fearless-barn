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

**The idea in one paragraph.** A base model predicts the taxi-out time of every departure from the
competition's own records. It is good on ordinary flights, but it cannot see what happened on the
apron: how long the aircraft waited after pushback, or how long it queued for the runway. For about half
of the departures an open archive of aircraft positions (ADS-B) shows exactly that. So a second, small
model looks at each departure's own ground track and corrects the base prediction. A few narrow rules
then cover cases neither model sees: weather, two airports with their own correctors, and a hard limit
from the flight plan.

Six stages. Stage 1 predicts every departure. Stages 2–6 correct that prediction where extra evidence
exists.

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

**How much each stage touches** (the final round file, 670,790 departures):

| stage | departures changed | typical size of the change |
|---|---:|---|
| 1. base model | all | the prediction itself; mean 1,006 s |
| 2. ADS-B corrector | 359,979 (53.7 %) | mean absolute change 85 s; 10th / 50th / 90th percentile −106 / −5 / +145 s |
| 3. weather offsets | 5,313 (0.8 %) | +30 s or +200 s |
| 4. LIRF corrector | 12,551 (1.9 %) | mean absolute change about 60 s |
| 5. LFPG corrector | 3,663 (0.5 %) | small |
| 6. LOBT window | 65 | large, on a handful of extreme rows |

**Two designs are named in this document:**
- **v61** is the best public board file. It has stages 1–5, with stage 4 in January only.
- **The final design** is what produced the final round file: v61's stages, with stage 4 in every
  month and stage 6 added. Both changes were tested for the final round (§5).

`final_pipeline.py` runs the final design by default; `--design v61` reproduces v61.

### 2.2 Stage 1: the base model

#### Inputs: 57 features

All features are built by `common/features.py`, identically for training and scoring. 42 are numeric and
15 are categorical.

| group | features | what they tell the model |
|---|---|---|
| calendar (10) | hour, minute of day, day of week, month, day of year, weekend flag, cyclic encodings of time of day and day of year | daily and seasonal rhythm of each airport |
| timing against the plan (12) | takeoff minus scheduled time, minus EOBT, minus IOBT, minus LOBT, and minus the NM actual off-block (AOBT); their absolute values and mutual differences | how late the flight was against each plan, and the NM's own view of when it left the stand |
| record flags (4) | has an NM flight record, has an AOBT, diverted plan, movement and flight type disagree | which rows are well documented and which are not |
| traffic (16) | departures in the previous and next 15 / 30 / 60 min; arrivals in the previous 15 / 30 / 60 min; same-runway departures before and after; mean taxi-in of recent arrivals; aircraft taxiing at pushback; a surge ratio | how busy the airport and the runway were around the takeoff |
| categoricals (15) | airport, runway, stand, destination and its region, aircraft type, operator, wake category, market segment, flight type, flight rule; the pairs stand × runway, airport × runway, airport × stand, airport × record flag | the fixed part of a taxi: where the aircraft starts, where it goes, who flies it |

**The single strongest input is takeoff minus the NM actual off-block.** The NM's off-block time is not
the airport's, and the two often differ by minutes, but it anchors the prediction. The model learns
the deviation from it.

**Not a forecast made at the gate.** All timings are measured **from the recorded takeoff**, the NM
**actual** off-block is an input, and some traffic counts include departures **after** the flight's
takeoff. The competition supplies these for every scored row (§9).

**Missing values.** Trees and CatBoost take them as they are. For the neural members, numeric gaps are
filled with the training median and all numerics are standardized; an unseen category maps to a
reserved "unknown" index.

#### The four model families

The families are deliberately different in kind. Their errors are only partly shared, and that is
what a blend can use.

| family | what it is | key settings | seeds |
|---|---|---|---:|
| **Trees** | LightGBM, one model per airport (10) and one per operator group (the 20 largest operators and "other"), blended with one global weight (0.68 on the per-airport model) | squared-error loss; learning rate 0.05; 255 leaves; at least 100 rows per leaf; 80 % feature and row sampling; 98–185 rounds per airport, frozen from early stopping | 1 |
| **MLP** | a feed-forward network on entity embeddings: each categorical gets an embedding of size √(number of levels), between 4 and 48; numerics are batch-normalized | layers 256-128-64, each Linear → BatchNorm → GELU → Dropout 0.1; AdamW, learning rate 1e-3, batch 2,048; squared-error loss; 10–19 epochs depending on the seed (14 for most) | 18 |
| **FT-Transformer** | a transformer over feature tokens: every feature becomes a token and attention lets features interact | four shapes: small (token width 64, 3 blocks), large (192, 3), deep (96, 6), wide (256, 2); 8 heads; AdamW; batch 1,024 | 6 / 9 / 6 / 6 |
| **CatBoost** | gradient boosting with ordered target statistics for categoricals | depth 8; learning rate 0.05; L2 3; Bayesian bootstrap; 2,500–7,400 trees depending on the seed | 6 |

- **Why trees per airport and per operator.** Taxi-out is first a property of the airport's layout,
  then of how an operator works there. On the inner holdout (August 2025) the per-airport model scored
  298.7, the per-operator model 308.3 and their blend 295.8.
- **Why many seeds.** A neural network trained twice gives two slightly different models. Averaging
  them removes that noise. Going from 1 to 18 MLP seeds was worth −29 on the board, the second largest
  gain of the project.
- **Why four FT-Transformer shapes.** Each shape is one quarter of the FT-Transformer slot. Different
  shapes disagree in useful ways; more seeds of one shape stopped paying.

#### The blend

Each airport has its own four weights (FT-Transformer, trees, MLP, CatBoost), summing to one:

| airport | FT-Transformer | trees | MLP | CatBoost |
|---|---:|---:|---:|---:|
| EDDF | 0.40 | 0.04 | 0.20 | 0.36 |
| EDDM | 0.29 | 0.27 | 0.32 | 0.12 |
| EGLL | 0.47 | 0.13 | 0.00 | 0.40 |
| EHAM | 0.76 | 0.15 | 0.00 | 0.09 |
| LEBL | 0.23 | 0.39 | 0.02 | 0.35 |
| LEMD | 0.25 | 0.25 | 0.25 | 0.25 |
| LFPG | 0.20 | 0.09 | 0.17 | 0.55 |
| LIRF | 0.00 | 0.00 | 1.00 | 0.00 |
| LSZH | 0.76 | 0.14 | 0.00 | 0.10 |
| LTFM | 0.64 | 0.07 | 0.00 | 0.30 |

- **How the weights are found.** For each airport, the weights that minimize squared error on
  `forward_inner` (models trained on January–June 2025, weights solved on July–August), constrained to
  be non-negative and sum to one. No scored model is fitted on that surface. The weights are then
  frozen (`tests_ftt_wide/final_weights_v59_s4.json`). At LEMD the solve stayed at equal weights.
- **LIRF is different.** The solver puts all of LIRF's weight on the MLP. LIRF then gets a second
  blend: 0.541 × MLP + 0.459 × TabNet (attentive feature selection; 5 seeds). LIRF holds most of the
  record artifacts (§1), and these two models handle them best. The 0.459 is the one constant fitted on
  the public board (§4).

**Target, clipping and refit:**
- **Target:** every valid row is kept, tail included. Dropping or capping the long taxi-outs made the
  board score far worse (§7).
- **Clipping:** predictions are clipped to [60, 86,400] s.
- **Final refit:** each member's number of rounds or epochs is first found by early stopping on a
  time-ordered split. The member is then retrained on all of 2025 for that fixed number, with no
  validation set to peek at.

**Two versions of this base appear in the results:**
- **v42:** three families (trees, MLP, three FT-Transformer shapes). Board 286.53. Versions v43–v57
  are built on it.
- **v59:** v42 plus the wide FT-Transformer shape and CatBoost as a fourth member
  (`tests_ftt_wide/PROBE.md`). It is the base of v58–v61 and of the final round file.

### 2.3 Stage 2: the ADS-B corrector

**What ADS-B adds.** Aircraft broadcast their position, speed and whether they are on the ground.
Volunteers' receivers record it, and adsb.lol publishes every day's recordings as open data. For a
departure whose aircraft was heard on the ground, the track shows when it started to move, where it
stood still, and when it lifted off. The base model has none of this.

**Data.** adsb.lol `globe_history` daily archives (ODbL): all of 2025, and January, February, June and
July 2026. About 1.2 TB were streamed for the first ranking set alone. Each archive is read once, and
only the matched points near the airports are kept (`tests_adsb_full/fetch.py`). The fetcher resumes
dropped connections and refuses to write a truncated day.

**Step 1: find the departure's aircraft** (`tests_adsb_full/features.py`).
1. **Candidates:** an aircraft that carried the departure's callsign (`FLIGHT_mvt` or
   `CALLSIGN_flt`) within 10 km of the airport, from 4 h before to 10 min after takeoff.
2. **Strict match:** the aircraft is kept only if its ground → air transition falls within ±120 s
   of the recorded takeoff. This rejects reused callsigns, inbound legs and wrong aircraft.

**Step 2: describe its taxi with 7 numbers.**

| feature | meaning |
|---|---|
| first sighting | seconds before takeoff of the first ground point |
| first displacement | seconds before takeoff when it had moved away from where it was first seen |
| first movement | seconds before takeoff of the first point moving at 3 knots or more |
| liftoff offset | observed liftoff against the recorded takeoff |
| stopped time | total seconds standing still during the taxi |
| last stop to liftoff | seconds from the end of the last stop to liftoff (the runway wait) |
| taxi speed | median ground speed while moving |

Three more candidates (number of sightings, path length, number of stops) shift between 2025 and 2026,
because the receiver network changed, so they are excluded. Features that describe *how* the aircraft
moved carry over between years; features that describe *how well it was heard* do not.

**Step 3: learn the base model's error from them.** A LightGBM model predicts `y − base`: how far the
base prediction was from the truth. Its inputs are the 7 features, the base prediction itself, the
airport and the hour.
- **Why the residual, not the taxi time.** The track covers only part of the taxi (see below), so it
  cannot replace the base model. It can say "this flight stood still for four minutes that the base
  model did not expect".
- **Held-out base predictions.** The corrector must learn the error the base model makes on flights
  it has **not** seen. A model's error on its own training rows is much smaller and would teach the
  corrector the wrong thing. So for every training month, the base prediction comes from base models
  trained without that month.
- **Training rows:** departures with a track at the seven airports, January and July–December 2025.
- **Size:** 3,000 trees, 127 leaves, learning rate 0.015; seeds 42–46 averaged
  (`tests_adsb_full/model_v61`).

**Coverage of the final set, by airport:**

| airport | departures corrected | mean absolute correction |
|---|---:|---:|
| EDDM | 94 % | 91 s |
| EHAM | 92 % | 85 s |
| LEBL | 91 % | 79 s |
| LSZH | 91 % | 81 s |
| EGLL | 83 % | 103 s |
| LEMD | 78 % | 75 s |
| EDDF | 38 % | 68 s |
| LFPG, LIRF | stages 5 and 4 | |
| LTFM | none | no usable ground coverage |

Departures without a track keep the base prediction.

**What it can and cannot see.** Receivers rarely see aircraft at the stand, so the first sighting
lags the recorded off-block by 4–11 min. ADS-B therefore informs the **taxi** (stops, the wait for
the runway, speed), not the off-block record.

### 2.4 Stage 3: weather offsets

The base model has no weather inputs, and the ADS-B corrector sees weather delays only through the
aircraft's own track. Departures **without** an ADS-B correction therefore get two fixed offsets,
from the last METAR within 90 min before takeoff:
- **January frost** (temperature ≤ 34 °F and temperature − dew point ≤ 3 °F): **+30 s**. Frost
  conditions mean de-icing, which adds time before the runway.
- **July thunderstorm** (`TS` in the weather codes): **+200 s**. Most of these rows are at LTFM.

**How much to trust them:**
- **Chosen in-sample.** Both values were hand-chosen on January + July 2025.
- **Mildly supported out of sample.** Fitted on 6 months and scored on the 7th, they give −0.07
  pooled (`../exploration/tests_weather_validated/`).
- **Scoped to their months.** Extending them to every month was tested for the final round and
  rejected (§5), so February and June get no weather offset.

### 2.5 Stages 4 and 5: single-airport ADS-B correctors

LIRF and LFPG are not in stage 2: LIRF's errors are dominated by record artifacts, and LFPG has very
few tracks. Each gets its own small corrector, built like stage 2 but for one airport.

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

The LOBT is the last off-block time filed in the flight plan. In 2025, every departure with a LOBT has
|off-block − LOBT| ≤ 3,606 s. Since taxi-out = takeoff − off-block, the taxi-out time must lie within
3,606 s of (takeoff − LOBT). Predictions are clipped to that window.

It is a property of the data, not a fitted model, and it only matters for a few extreme predictions:
it moves 65 of the 670,790 final rows. The idea is from team outgoing-boat (§11); it was re-derived on
our data and tested for the final round (§5).

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

**How stage 1 itself was built** (board, before any ADS-B):

| step | board | change |
|---|---:|---:|
| one pooled LightGBM | 405.75 | |
| trees per airport | 386.22 | −19.5 |
| + one MLP, blended per airport | 324.28 | −61.9 |
| MLP seeds 1 → 18 | 295.21 | −29.1 |
| + TabNet at LIRF, + FT-Transformer | 289.78 | −5.4 |
| FT-Transformer seeds and three shapes | 286.53 | −3.3 |

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
  `../exploration/tests_ctx/`) passed on Feb–Jun but was +0.44 worse on January + July 2025, with almost all of its
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

Every row was tested; each has a `RESULT.md` in its `tests_*` folder (most are in `../exploration/`).

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

The repository has two folders:
- **`solution/`** (this folder): the final pipeline, the code that trains its models, and the saved
  models and files it reads.
- **`../exploration/`**: every other experiment, the early analysis scripts and the decision history.

Paths in this document are relative to `solution/`.

**`solution/`**

| path | contents |
|---|---|
| `final_pipeline.py` | the whole recipe, stages 1–6 |
| `predict_new.py` | stage 1 from saved models, for any movements file |
| `submit.py` | assembles base-model predictions for the ranking set; validates a submission file |
| `common/`, `config.py` | data loading, features, splits |
| `pipeline_*/` | training code of the base-model members (8 folders) |
| `artifacts/` | metadata of every trained run (parameters, metrics, splits); the model files are not in git (§8) |
| `submissions/` | the final round file, key board files, `board_scores.csv` (every score with its pre-written expectation) |
| `requirements.txt` | Python dependencies |

**Where each stage lives.** The `tests_*` folders kept in `solution/` hold a stage's code and saved
model together with the experiments that selected it.

| stage | code and tests | saved model or weights |
|---|---|---|
| 1. base model | `pipeline_per_airport/`, `pipeline_per_operator/`, `pipeline_airport_operator_blend/`, `pipeline_mlp_antigravity/`, `pipeline_ftt_multimodel/`, `pipeline_catboost_antigravity/`, `pipeline_tabnet_lirf_override/`; `tests_ftt/`, `tests_ftt_wide/`, `tests_tabnet/` | `tests_ftt_wide/final_weights_v59_s4.json`; trained models not in git (§8) |
| 2. ADS-B corrector | `tests_adsb_full/` (fetch, matching, features, rounds 1–7); size and training base in `../exploration/tests_round10/` – `tests_round13/` | `tests_adsb_full/model_v61/` |
| 3. weather offsets | `../exploration/tests_weather_metar/`, `../exploration/tests_weather_validated/` | two constants in `final_pipeline.py` |
| 4. LIRF corrector | `tests_lirf_adsb/` | `tests_lirf_adsb/model_v55/` |
| 5. LFPG corrector | `tests_round9/` | `tests_round9/model_v56/` |
| 6. LOBT window | `tests_final/` | a rule in `final_pipeline.py` |
| final round build, reproducibility | `tests_final/`, `tests_repro/` | `tests_final/final_base.parquet` |

**`../exploration/`** (indexed in `../exploration/README.md`)

| path | contents |
|---|---|
| `EXPLORATION_SUMMARY.md` | one-page list of everything tried |
| `tests_*/` | 58 experiments, each with its pre-registration and result; among them `tests_copymix/`, `tests_lirf_spec/` and `tests_ctx/` (ideas from other teams, tested and rejected) and `tests_febjun/` (the Feb–Jun held-out base) |
| `pipeline_*/`, `overtuned/` | 26 candidate base models that were not selected, and the ones quarantined as over-tuned |
| `stage*.py`, `scratch_*.py`, `compare*`, `run_*`, `logs/` | early analysis scripts and their outputs |
| `plans/`, `PIPELINE_REVIEW.md`, `TAXI_OUT_DRIVERS.md` and the other `.md` files | decision history and early analyses, not maintained |

**A note on the experiment scripts.** They were written when both folders were one. After the split,
a small helper (`tools/repo_layout.py`) lets a script in either folder find files in the other. The
final pipeline was re-verified after the split (§8). The experiment scripts were updated mechanically;
a sample was re-run and reproduced its earlier numbers, but not all of them were re-run.
