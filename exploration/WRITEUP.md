# fearless-barn — PRC Data Challenge 2026: taxi-out time

| | |
|---|---|
| **Final round** | `fearless-barn_final.parquet`: 670,790 departures (Jan, Feb, Jun, Jul 2026), built by `final_pipeline.py` with the final design (§12) |
| **Best board score** | **v61 = 269.9888** pooled RMSE on the public board (ranking uses each team's best submission) |
| **Without ADS-B** | **v42 = 286.5256**, the base model alone (competition data only), kept as a byte-identical file (§9) |
| **Standing** (2026-10-08) | #59 of 120 on the public board (on 2026-10-03: #51 of 221 by best score; leader then 219.64, #10 230.37, median 298.8) |
| **Reproduce / predict** | `python final_pipeline.py ...`, verified byte-identical to the uploaded v56 and v61 (`--design v61`), and verified on new data (§10, §12) |
| **Uploads** | 45 on the leaderboard record (from 2026-09-08), all distinct files; every upload a full 344,841-row file, its expected score written down before upload (§7) |

---

## 1. Task, data and metric

- **Task:** predict the taxi-out time (seconds from off-block to takeoff) of every departure in the
  scored set: **January and July 2026** at ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL,
  LEMD, LFPG, LIRF, LSZH, LTFM).
- **Training data:** all of 2025, 2,084,659 labelled departures with their movement and flight
  records.
- **Metric:** pooled RMSE over 344,841 rows.

**The metric shapes the whole design.** About 0.02 % of departures (taxi-outs over 2 hours) carry
15–56 % of the squared error, depending on the period. Most of them are record artifacts at Rome:
a missing NM flight record, and an off-block time copied from the schedule. Under squared error,
the best prediction is the conditional **mean**, so every method that shrinks the tail
(winsorizing, robust losses, prediction caps) loses (§8).

## 2. Solution overview

```
 movements file ──► 1. BASE MODEL  (v42; the competition's data only)
                       per-airport blend of trees + MLPs + FT-Transformers, TabNet override at LIRF
                    │
 adsb.lol archive ──► 2. ADS-B CORRECTOR  (7 airports: EDDF EHAM LEBL EDDM LSZH LEMD EGLL)
                       LightGBM on the base's residual, from the departure's own ground track
                    │
 METAR ─────────────► 3. WEATHER OFFSETS  (departures without an ADS-B correction)
                       frost in January +30 s; thunderstorm in July +200 s
                    │
 adsb.lol archive ──► 4. LIRF CORRECTOR  (January; departures with an NM record and a track)
                    │
 adsb.lol archive ──► 5. LFPG CORRECTOR  (departures with an NM record and a track)
                    │
 flight plan (LOBT) ► 6. LOBT WINDOW  (final design only, §12)
                    ▼
                    prediction (v61 on the board; the final design for the final round)
```

**Board score after each stage:**

| stage | board |
|---|---:|
| 1. base model | 286.53 |
| + 2. ADS-B corrector | 272.17 |
| + 3. weather offsets | 271.87 |
| + 4. LIRF corrector | 271.76 |
| + 5. LFPG corrector | 271.56 |
| larger stage-2 corrector (1,500 trees, 63 leaves; v57) | 270.88 |
| base + wide FT-Transformer (d256×2) + CatBoost as a 4th member (v58) | 270.81 |
| stage-2 corrector at 3,000 trees / 127 leaves (v60) | 270.47 |
| stage-2 corrector trained on residuals of a realistic, per-airport weighted base (v61) | **269.99** |

## 3. Stage 1: the base model (v42)

**Features** (57, the `with_aobt` featureset, built identically for training and scoring). Several use
information the competition supplies for every scored row but that a forecast made at off-block would
not have: all timings are measured **from the recorded takeoff**, the NM **actual** off-block (AOBT) is
an input, and some traffic counts include departures **after** the flight's takeoff.
- **Calendar and time of day:** hour, day of week, day of year, cyclic encodings.
- **Timing from the flight plan and the NM flight record,** relative to takeoff: scheduled, EOBT,
  IOBT, LOBT and NM actual off-block (AOBT), their differences, and flags for missing records.
- **Categoricals:** airport, runway, stand, destination, aircraft type, operator, wake category,
  market segment, flight type and rules, plus the interactions stand × runway, airport × stand and
  airport × runway.
- **Traffic from the movement records:** departures and arrivals in 15-, 30- and 60-min windows,
  same-runway departures, recent taxi-in times, aircraft taxiing at pushback, and a surge ratio.

**Model:**

```
                    ┌─ trees:  per-airport + per-operator LightGBM, one global blend weight
per-airport blend ──┼─ MLP:    entity-embedding MLP (256-128-64), mean of 18 seeds
 (3 weights per     └─ FT-Transformers, one third each of three shapes:
  airport)               small d64×3 blocks (6 seeds), large d192×3 (9 seeds), deep d96×6 (6 seeds)
        └─ at LIRF only:  0.541 × blend + 0.459 × TabNet (mean of 5 seeds)
```

- **Per-airport blend weights:** solved on `forward_inner` (train Jan–Jun 2025, held out Jul–Aug),
  a surface no scored model is fitted on, then frozen. At LIRF the FT-Transformer weight solves to
  0 every time.
- **Target and clipping:** every valid row is kept, tail included (`keep_all`); predictions are
  clipped to [60, 86,400] s.
- **Final refit:** all members are refit on the full year with the hyperparameters and epoch counts
  frozen from time-ordered splits.

**What built it, in order of board gain** (§6 has the full history):
- blending a neural network with the trees: −62;
- MLP seed averaging (1 → 18): −29;
- architecturally different neural members (TabNet at LIRF, FT-Transformers elsewhere): −5.4;
- FT-Transformer seeds and three FT-Transformer shapes: −2.9.

Inside the MLP family, no variant ever paid.

## 4. Stages 2, 4 and 5: ADS-B correctors

**Data:**
- **Source:** adsb.lol `globe_history` daily archives (open data, ODbL), for all of 2025 and
  Jan + Jul 2026.
- **Volume:** about 1.2 TB streamed; each archive is read once and only the matched points near the
  airport are kept (`tests_adsb_full/fetch.py`).
- **Integrity:** the fetcher resumes dropped connections and refuses to write a truncated day (ten
  silently truncated days were found and refetched).

**Matching and features** (`tests_adsb_full/features.py`):
1. **Candidates:** an aircraft that carried the departure's callsign (`FLIGHT_mvt` or
   `CALLSIGN_flt`) within 10 km of the airport, from 4 h before to 10 min after takeoff.
2. **Strict match:** the aircraft is kept only if its ground → air transition falls within ±120 s
   of the recorded takeoff. This rejects reused callsigns, inbound legs and wrong aircraft.
   Coverage: 44–62 % of departures in 2025–26; LFPG ≈ 5 %, LTFM 0 %, LIRF Jan 2026 63 %, Jul 2026 0 %.
3. **The 7 features (`all_nd`):**
   - seconds before takeoff of the first ground sighting, first displacement from the stand and
     first movement;
   - the liftoff offset;
   - total stopped time;
   - time from the last stop to liftoff;
   - median taxi speed.
4. **Drift gate:** three more candidate features (sightings count, path length, number of stops)
   shift between 2025 and 2026, because the receiver network changed, so they are excluded.
   Features that describe *how* the aircraft moved carry over between years; features that
   describe *how well it was heard* do not.

**Correctors:** a LightGBM model on the residual `y − base` of **held-out** base predictions, that
is, predictions from models that never saw the month in question. Seeds 42–46 are averaged.

| stage | where | trained on | rows corrected (2026) |
|---|---|---|---:|
| 2 | EDDF, EHAM, LEBL, EDDM, LSZH, LEMD, EGLL; departures with a track | Jan, Jul–Dec 2025; 3,000 trees, 127 leaves, trained on residuals of a base weighted per airport like the shipped blend (v61; v60 used an equal-weight training base; v57–v59 1,500 / 63, v56 600 / 31) | 188,635 (54.7 %) |
| 4 | LIRF, **January**; departures with an NM record and a track | Jul–Dec 2025 | 6,960 |
| 5 | LFPG; departures with an NM record and a track | Jul–Dec 2025 (Jan 2025 has no LFPG tracks) | 1,973 |

**What it can and cannot see:**
- Receivers rarely see aircraft at the stand, so the first sighting lags the recorded off-block by
  4–11 min. ADS-B therefore informs the **taxi** (stops, the wait for the runway, speed), not the
  off-block record.
- **This explains LIRF:** a corrector trained on all LIRF rows failed (Aug 2025: 654 → 675),
  because the no-record rows are record artifacts. Restricted to departures with a record, it
  passed (−4.2 RMSE on LIRF, 5 of 6 months).

**How each step was accepted.** Each was confirmed on a fresh 2025 season before its upload:

| version | change | local test | board |
|---|---|---|---:|
| v43 | 3 airports | Jan ↔ Jul plus 44 fresh days: 44/44 days better | 281.86 (−4.66) |
| v44 | + EDDM | Nov–Dec 2025, frozen corrector: 61/61 days | 279.15 (−2.71) |
| v45 | + LSZH, LEMD, EGLL; + Nov–Dec training | Sep–Oct 2025: 61/61 days | 273.60 (−5.55) |
| v46 | + Sep–Oct training; 600 trees | Aug 2025: 31/31 days | 272.26 (−1.34) |
| v48 | 5-seed average; + Aug training | leave one month out, 7/7 months | 272.17 (−0.09) |
| v55 | LIRF corrector (record rows) | leave one month out, 5/6 months | 271.76 (−0.11) |
| v56 | LFPG corrector (record rows) | leave one month out, 6/7 months | **271.56 (−0.20)** |

## 5. Stage 3: weather offsets

The base model has no weather inputs. The ADS-B corrector sees weather delays only through the
aircraft's own track, so departures **without** an ADS-B correction get two fixed offsets, using the
last METAR within 90 min before takeoff:

- **January frost** (tmpf ≤ 34 °F and tmpf − dwpf ≤ 3 °F): **+30 s** (4,878 departures; board
  v49 −0.23);
- **July thunderstorm** (`TS` in wxcodes): **+200 s** (428 departures, 370 of them at LTFM; board
  v50 −0.07).

**How much to trust them:**
- **Chosen in-sample:** both values were hand-chosen on Jan + Jul 2025 and first measured on the
  same data. At this size the local estimate did not even rank the two correctly.
- **Mildly supported out of sample:** a later out-of-sample check (offsets fitted on 6 months,
  scored on the 7th) supports them mildly: −0.07 pooled.
- **The fitted alternative is not better on the scored months:** per-condition offsets fitted out
  of sample are −0.16 pooled, but worse in January.
- **Per-airport frost offsets were not supported.** The four candidates built on them (v51–v54)
  were never uploaded.

## 6. Results: board history

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
| v43 → v48 | ADS-B corrector, 7 airports | 281.86 → 272.17 | −14.36 |
| v49, v50 | weather offsets | 271.94, 271.87 | −0.30 |
| v55 | LIRF ADS-B corrector | 271.76 | −0.11 |
| v56 | LFPG ADS-B corrector | 271.56 | −0.20 |
| v57 | larger 7-airport corrector (round 10) | 270.88 | −0.68 |
| v58 | base: + wide FT-Transformer, + CatBoost family | 270.8103 | −0.07 |
| v59 | wide FT-Transformer and CatBoost 3 → 6 seeds | 270.81 | −0.0003 (tie) |
| v60 | 7-airport corrector at 3,000 trees / 127 leaves (round 11) | 270.47 | −0.34 |
| **v61** | **7-airport corrector trained on the realistic (per-airport weighted) base (round 12)** | **269.99** | **−0.48** |

**Uploads that lost.** The ranking page ranks each team on its best submission, so a losing upload costs only a daily slot:

| version | change | result |
|---|---|---:|
| v18 | prediction cap at 2 h | 491.35 |
| v27 | target winsorized at 2 h | 496.09 |
| v15, v17 | LIRF identity override | +0.86 twice |
| v33 | TabNet 11 seeds | +0.91 |
| v37 | season-weighted MLPs | +2.07 |
| v47 | tabular columns inside the ADS-B corrector | +0.47 |
| v63 | copy-mixture member (two-stage idea from outgoing-boat, §13) + LOBT-window clip | 270.2 (+0.21) |

## 7. Validation protocol and how the leaderboard was used

**Splits.** All splits are time-ordered; nothing is selected on the full-year refit.

| split | train | held out | used for |
|---|---|---|---|
| `forward` | Jan–Aug | Sep–Oct (val), Nov–Dec (test) | early stopping, comparison |
| `forward_inner` | Jan–Jun | Jul–Aug | blend weights, solved out of sample |
| `seasonal_full` | Feb–Jun, Aug–Nov | Dec (val), **Jan + Jul** (test) | the board's months; confirmation before any build |
| `full` | all of 2025 | — | the final refit only |

**Rules applied to every experiment:**
- **The pass bar is written before the result exists.** Every test has a `PREREGISTERED*.md`,
  committed before it ran, with its decision rule and the action for pass or fail.
- **Confirmation on data the choice never saw.** ADS-B rounds were each confirmed on a fresh 2025
  season; later rounds used leave-one-month-out.
- **Controls:**
  - permutation controls (the new inputs shuffled) must lose;
  - the result must survive removing the 5 largest-gain rows, because the tail can fake a gain;
  - several seeds, paired by seed.
- **Drift gate for any external input:** 2025 against 2026 in the same months; a standardized
  shift above 0.2 disqualifies the feature.
- **A training run that hit its epoch cap is unresolved, not failed.** A convergence audit re-read
  every early verdict.

**Use of the public leaderboard:**
- **45 uploads** on the leaderboard record (from 2026-09-08), all distinct files. Each was a
  full prediction file with its expected score and fallback written down beforehand. Most are new
  models; v49, v50, v55 and v56 instead add corrections to defined subsets of departures (weather
  conditions, LIRF and LFPG tracked flights). No upload altered single rows to read their labels.
- **What a subset correction reveals, and why it was not used.** Adding a constant to a fixed subset
  and reading the score change reveals that subset's mean error on the hidden labels. v49's result
  implies the January frost subset is under-predicted by about 160 s on average, against the +30 s
  applied. That number was deliberately **not** used to set any later value.
- **One constant was fitted on the board:** the LIRF TabNet weight 0.459. LIRF's prediction is
  linear in that weight, so board MSE is a parabola through three existing uploads; v35 landed
  0.0001 s from the predicted optimum.
- **The weather offsets** (§5) were hand-chosen, which §5 states openly.
- **v51–v54 were not uploaded:** four weather-offset variants were built late, and uploading
  several to keep the best would have tuned to the hidden labels. They sit in
  `submissions/DO_NOT_UPLOAD/`.
- **Local results under-read the board for averaging and model variety:**
  - MLP 1 → 3 seeds: −0.9 locally, −18 on the board;
  - FT-Transformer 3 → 6 seeds: −0.2 locally, −1.8 on the board.

  For these questions a single upload with a fallback was the only usable measurement, and it was
  used that way.
- **Local results over-read the board for tail gains (v63).** The copy-mixture test passed its
  pre-registered bar (−1.01 pooled, 6/7 months), but the reference it was compared against lacked
  the LOBT-window clip that the candidate carried, so the clip's gain was credited to the new member.
  Against the clipped reference the member was +0.22 worse on January + July, and the board gave
  +0.21. Two rules follow:
  - the reference in a test must carry every fixed rule the candidate carries;
  - a gain is judged on the board's months and on the body (y ≤ 1 h), not on the tail. outgoing-boat
    report the same finding from six submissions (body share predicts the board change with
    correlation 0.985; the tail share is anti-correlated).

## 8. What did not work (all tested, records in `tests_*/RESULT.md`)

| idea | result |
|---|---|
| winsorize / cap / floor the target or the predictions | large losses (board 491–496) |
| log target, Huber, Poisson, quantile ladder, hurdle model | all de-weight the tail and lose |
| LIRF identity override (no record → scheduled delay) | right in 2025, +0.86 on the 2026 board twice |
| sequence and neighbour models (GRU, queue attention, retrieval) | residuals 0.98 correlated with the MLP |
| MLP variants, PLE embeddings, TabM, season weighting, transductive pretraining | all worse |
| CatBoost and XGBoost as further members; tree-side seeds | no gain once blended |
| METAR weather as model features; holidays; OSM stand distance | within noise |
| runway queue / runway-mode features from the records | stable across years but +2.1 worse as a corrector |
| routed taxi distance (apt.dat), measured as a ceiling | at most −0.14 s, not built |
| ADS-B features inside the tree model | tree −7 alone, +0.08 once corrected |
| traffic counts from other aircraft's ADS-B | fail the drift gate (they track receiver coverage) |
| tabular columns inside the ADS-B corrector | 7/7 months locally, +0.47 on the board |
| Feb–Jun as extra corrector training (new held-out base) | +0.07, 2/7 months |
| looser ADS-B matching at EDDF/LEMD | the extra matches have no ground points |
| stage-2 corrector trained on a 4-member (v59-weighted) base (round 13) | 4/7 months, FAIL |
| LFPG corrector retrained on the 4-member base (round 13) | FAIL |
| copy-mixture member: copy classifier + normal regressor, blended per airport (v63) | Jan + Jul +0.22 against the clipped recipe; board +0.21 |
| per-airport line on delay-to-schedule for no-record rows over 6 h (outgoing-boat's rule) | −12.7 locally, but all from ~10 LIRF tail rows; worse once the 20 best rows are removed; not shipped |
| LOBT-window clip alone (v62: 45 ranking rows) | true for 100 % of 2025 departures; built, not uploaded (a 45-row upload is too close to probing) |

**Two lessons stand out:**
- **An idea closed on a weak test may be wrong.** ADS-B was first "closed" on three days and one
  crude feature, and later became the largest gain since the first neural model.
- **A within-2025 holdout cannot see a 2025 → 2026 shift.** v47 passed locally and lost on the
  board, and the drift gate exists because of it.

## 9. Design trade-off: post-pushback ADS-B

Stages 2, 4 and 5 use each departure's **own** ADS-B ground track after pushback: when it moved,
how long it stopped, how fast it taxied. A real-time forecast made before off-block would not have
this.

- **What it gives:** −15.0 s on the board (286.53 → 271.56).
- **What it costs:** it is not available before off-block. It relies on volunteer receiver
  coverage, which varies by airport and changed between 2025 and 2026. It describes the physical
  taxi, not the recorded off-block.
- **Status:** adsb.lol is open data, and the organizers wrote that open data sources can be used.
  A forum question asked whether post-pushback ADS-B counts as leakage, and no explicit answer is on
  record.
- **What is new and what is not.** The competition's own inputs are already partly after the
  event: the recorded takeoff time, the NM actual off-block, and later departures' times (§3). What
  ADS-B adds is the aircraft's **own observed taxi**, which is the closest any input comes to the
  quantity being predicted.
- **v42 is the model without ADS-B.** It uses only the competition's data, and it is kept as a
  byte-identical file. If post-pushback ADS-B is not accepted, v42 is the submission.

## 10. Reproducibility

**One command, any movements file in `ranking.parquet`'s schema:**

```bash
python final_pipeline.py --movements new.parquet --points-dir adsb_new --fetch --out pred.parquet
```

**What the command does:**
1. runs the base model from the saved models (`predict_new.py`);
2. streams the adsb.lol archives for the departure days (`tests_adsb_full/fetch.py`);
3. fetches any METAR reports the local weather file lacks, from the Iowa Environmental Mesonet;
4. applies stages 2–5 with the saved correctors.

**Verified:**
- **The upload:** on the competition's ranking set, from the saved v42 base, the output equals the
  uploaded **v56 byte for byte** (344,841 rows):
  `python final_pipeline.py --base submissions/fearless_barn_v42.parquet --check submissions/fearless_barn_v56.parquet --stage2-model tests_adsb_full/model_v48 --out final.parquet`;
  and the final **v61 byte for byte** from the v59 base:
  `python final_pipeline.py --base tests_ftt_wide/v59_base.parquet --check submissions/fearless_barn_v61.parquet --out final.parquet`
  (`--s2-out` also writes the stage-2 output before stages 3–5).
- **New data:** three 2025 days treated as new data (15,408 departures), run from scratch with
  fresh ADS-B downloads, give output identical to the run on the existing ADS-B files. The METAR
  fetch was checked on dates after the local weather file ends.
- **The base from raw data:** `predict_new.py` rebuilds the v42 base from raw `ranking.parquet` on
  344,840 / 344,841 rows (one LIRF row off by 1 s, a TabNet reload rounding boundary).
- **Every ADS-B candidate (v43–v61)** rebuilds byte-identical from its saved models.

**Scope of the rules.** The weather offsets apply only in January (frost) and July (thunderstorms),
and the LIRF corrector only in January, as built for the scored months. On data from other months
those stages change nothing; the ADS-B corrector at the 7 airports and the LFPG corrector apply in
any month.

**What git does not hold:**
- the trained runs (`artifacts/runs/`);
- the feature cache;
- the FT-Transformer and TabNet checkpoints;
- the ADS-B day files (`tests_adsb_full/points/`).

Two sha256 manifests list them:
- `tests_repro/dependency_manifest.csv`: the base model's files (runs, feature cache, checkpoints);
- `tests_repro/dependency_manifest_v56.csv`: what stages 2–5 add (the 2026 ADS-B day files and the
  corrector models).

The ADS-B day files (2.9 GB) can be re-fetched; the trained runs (about 37 GB) and checkpoints
cannot. **They are the backup set.**

## 11. Data sources and licences

| data | use | source / licence |
|---|---|---|
| 2025 training movements and flights, 2026 ranking set | everything | PRC Data Challenge 2026 (EUROCONTROL / airports) |
| METAR, 10 airports | stage 3 | downloaded by us from the Iowa Environmental Mesonet ASOS archive (public) into `prc-2026-datasets-wx/`; IEM again for any gap |
| ADS-B ground tracks | stages 2, 4, 5 | **adsb.lol** `globe_history`, © adsb.lol contributors, **ODbL** |

The OpenSky historical database was not used. OPDI (opdi.aero) was confirmed as open data by the
organizers on the forum (2026-10-08), but it was deliberately not used: its ground coverage is
weakest at the airports where our ADS-B is also missing (LFPG, EGLL, LIRF, EDDM, LTFM), and it
arrived too late to test before the freeze.

## 12. The final round (blind; Jan, Feb, Jun, Jul 2026)

On 2026-10-08 the organizers added a blind final ranking: one file for `final_submitting.parquet`
(670,790 departures: January, February, June and July 2026), scored on the first two months, the two
new months and all four, with no leaderboard feedback. The design freezes with the documentation
(2026-10-11); the file is due 2026-10-14 12:00 CEST.

**Re-checking the design on the new months.** Several stages were scoped to the first ranking set's
months (frost: January; thunderstorm: July; LIRF corrector: January, because LIRF had no ADS-B in July
2026). Each was re-checked on **held-out Feb–Jun 2025**, with the saved correctors (none trained on those
months) and a pre-registered bar (`tests_final/PREREGISTERED.md`):

| | Feb–Jun 2025 | Feb + Jun 2025 |
|---|---:|---:|
| base | 285.11 | 327.74 |
| shipped design (stage 2 + LFPG corrector) | 274.13 | 317.39 |

The ADS-B corrector transfers to the new months (−11.0, 5/5 months, −11.4 with the best 20 rows removed).

| change against the shipped design | rows | Feb–Jun | Feb + Jun | months | best 20 removed | decision |
|---|---:|---:|---:|---:|---:|---|
| LIRF corrector in every month | 13,009 | −0.30 | −0.47 | 2/2 | −0.19 | **adopted** |
| LOBT-window clip | 86 | −0.11 | −0.11 | 5/5 | −0.005 | **adopted** |
| frost +30 s in every month | 3,305 | −0.14 | −0.31 | 1/3 | −0.12 | rejected |
| thunderstorm +200 s in every month | 827 | −0.06 | −0.15 | 3/5 | −0.005 | rejected |
| LIRF no-record specialist (quirky-honey's idea) | 470 | −15.8 | −32.0 | 4/5 | +3.29 | rejected |
| the same, only > 6 h behind schedule | 14 | −14.3 | −24.5 | 4/5 | +0.009 | rejected |

**The specialist was a judgement call, disclosed here.** It also failed on the seven scored months
(`tests_lirf_spec/RESULT.md`): −14.0 pooled, but +2.5 with the best 20 rows removed. Its hybrid form
improved 11 of the 12 months of 2025, yet its whole gain sits in about 20–30 LIRF departures a year
with no flight record and a takeoff many hours behind schedule. That is the kind of gain that has failed
to transfer before (our LIRF identity override, outgoing-boat's v37), and it failed our pre-registered
robustness bar. We kept to the bar and left it out.

**The final design** = v61's stages, with the LIRF corrector in every month and the LOBT clip:
- `final_pipeline.py` (default `--design final`);
- the base is rebuilt from saved models by `predict_new.py --weights tests_ftt_wide/final_weights_v59_s4.json`,
  which reproduces the v59 base on the ranking set on 344,840 / 344,841 rows (one LIRF row off by 1 s);
- the ADS-B day files for February and June 2026 (59 days) were fetched with the same `fetch.py`.
  Coverage of departures with track points: Jan 81 %, Feb 84 %, Jun 77 %, Jul 83 %.

**Build:**
1. `python tests_final/rebase_points.py` writes `tests_final/points_final/` (see below);
2. `python final_pipeline.py --movements ../prc-2026-datasets/final_ranking.parquet --base tests_final/final_base.parquet
   --points-dir tests_final/points_final --out submissions/fearless-barn_final.parquet`, where `final_base.parquet` is
   `predict_new.py --weights tests_ftt_wide/final_weights_v59_s4.json` on the same movements file.

Result:
- 670,790 rows, same IDs and order as the template;
- ADS-B corrected 359,979; frost 4,883; thunderstorm 430; LIRF corrector 12,551; LFPG 3,663;
  LOBT clip 65.

**The final file's January and July rows are not the first ranking set's rows.** In `final_ranking.parquet`:
- takeoff times differ from `ranking.parquet` on 92 % of rows (median 4 s, at most 12 s);
- about 1 % of flight-record fields differ;
- scheduled times are unchanged.

Time-of-day, delay and traffic-window features move accordingly, so the final file's January and July
predictions differ from v61's (RMS 9.9 s). They are predicted from the final file's own data, as the task
requires.

**ADS-B times re-based for the same reason.** Each stored ADS-B point carries its time relative to the
movement's takeoff time in the file it was matched against.
- The January and July day files were matched against `ranking.parquet`, so `tests_final/rebase_points.py`
  re-bases them on `final_ranking.parquet`'s takeoff times (316,809 movements shifted, by up to 12 s).
- 2025-12-31 and 2026-01-31 were re-fetched against `final_ranking.parquet`. They hold the pre-midnight ground
  points of early takeoffs on 1 January and 1 February.
- February and June were fetched against `final_ranking.parquet` from the start.
- Effect on the final file: January rows RMS 11.7 s, July 8.1 s, mean 0.0.

**Limitation of the Feb–Jun check.** At LIRF the check used the 3-member held-out base. The shipped LIRF
base adds the TabNet override, which has no held-out Feb–Jun predictions. The LIRF corrector decision was
therefore made on a slightly different base than the one it is applied to.

## 13. Comparison with a published solution (outgoing-boat)

Team outgoing-boat published their solution (github.com/EnioAguiar/prc-taxiout-2026, GPL-3.0; best
board 243.95). We read it after our model was built, for ideas; none of their code is used. Two of
their ideas were re-implemented and tested by us (v63, §8; the LOBT clip is in the final design, §12) and are credited to them: the copy
classifier with an expectation mixture, and the LOBT window (|BLOCK − LOBT| ≤ 3,606 s for every 2025
departure).

**Where the 26 s gap comes from, by each team's board history:**

| | outgoing-boat | fearless-barn |
|---|---:|---:|
| model without ADS-B | 331 | **286.5** (v42) |
| + ADS-B, cross-fitted corrector, LOBT window | 276 (−55) | 272.2 (−14.4) |
| + context and external data in the corrector | 248 (−28) | 271.6 (−0.6: weather offsets, Rome/Paris correctors) |
| + more correctors, rebuilt base | 244 (−4) | 270.0 (−1.6: corrector capacity, realistic training base) |

**Our base model is about 45 s better than theirs; they lead because of what their corrector sees
and which rows it reaches.**
- **Coverage.** Their corrector runs on every row at all ten airports, with ADS-B fields as inputs
  that may be empty. Ours runs only on departures with a matched ground track at seven airports,
  plus LIRF and LFPG models: about 55 % of rows.
- **Inputs.** Their corrector reads the airport's state at the time:
  - arrivals' taxi-in per runway and the last arrival at the stand;
  - runway configuration and its changes, and the same-runway queue weighted by wake category;
  - gate retention (departures past EOBT not yet pushed);
  - EUROCONTROL daily delay and regulation series, OPDI ground time since the last landing;
  - per-airline copy rates;
  - X-Plane stand-to-runway distances.

  Ours reads the departure's own track and the base prediction. We tested several of these as
  base-model features (runway mode, queue, routed distance, ATFM delay) and they did not pay there
  (§8). In their design they enter the corrector, trained against an out-of-block base.
- **Design around the label.** Their base splits "off-block copied from the schedule" (about 9 % of
  departures) from real taxi times. Ours learns both in one model. Bolted on at the end (v63), the
  split did not pay, because our corrector already covers part of the same signal.
- **Validation.** They decide on January + July 2025 and on the body of the distribution. That is
  the rule our v63 result argues for (§7).

**What we would do with more time:**
1. run the corrector on every row with all-row inputs (arrivals, runway state, gate retention);
2. keep our stronger base under it.

Neither was feasible in the days left.

## 14. Where things are

| path | contents |
|---|---|
| `final_pipeline.py` | the whole recipe (§10) |
| `predict_new.py` | stage 1 from saved models |
| `tests_adsb_full/` | ADS-B fetch, matching and features; rounds 1–7 (`RESULT.md`, `PREREGISTERED*.md`) |
| `tests_lirf_adsb/`, `tests_round9/` | the LIRF (v55) and LFPG (v56) correctors |
| `tests_weather_metar/`, `tests_weather_validated/` | the weather offsets and their out-of-sample check |
| `tests_*/` | every other experiment, each with its pre-registration and result |
| `tests_round10/`–`tests_round13/` | stage-2 corrector capacity (v57, v60), realistic training base (v61), 4-member base (failed) |
| `tests_copymix/` | copy-mixture member (v63), the v63 build, the outgoing-boat rule check |
| `tests_lirf_spec/` | LIRF no-record specialist and its hybrid (rejected) |
| `tests_final/` | the final round: Feb–Jun re-check, ADS-B fetch for Feb and Jun 2026, the final build logs |
| `submissions/board_scores.csv` | every board score with its pre-written expectation |
| `EXPLORATION_SUMMARY.md` | one-page list: done, explored, not explored |
| `plans/`, `PIPELINE_REVIEW.md`, `TAXI_OUT_DRIVERS.md` | decision history and early analyses |
