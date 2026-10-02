# fearless-barn — what we did, what we explored, what we did not

PRC Data Challenge 2026: predict taxi-out time (off-block → takeoff) for January and July
2026 departures at ten European airports, scored by pooled RMSE. Status on 2026-10-01.
Full method: `WRITEUP.md`. Evidence for every line below: the named `tests_*/RESULT.md`,
`TAXI_OUT_DRIVERS.md` section, or `plans/` file.

## 1. Where we ended up

| | |
|---|---|
| Final model | **v56 = 271.5625** (v55 + LFPG ADS-B corrector); v55 = 271.764 (v50 + LIRF ADS-B corrector); before them v50 = 271.8748 on the public board (from 405.75 with the first submission); clean no-ADS-B fallback v42 = 286.53 |
| Standing | **#46 of 214** teams by best score (was #70 before ADS-B); leader 219.64, #10 233.8, #50 273.6, median 298.9 |
| Model | per-airport blend of LightGBM trees + 18 MLPs + 21 FT-Transformers (three architectures), plus TabNet at LIRF, 5-seed ADS-B corrector, and winter morning de-icing offset |
| Reproducibility | **`final_pipeline.py` rebuilds the final v50 byte-identical** (base → ADS-B corrector → weather offsets); `predict_new.py` rebuilds the v42 base from raw data on 344,840 / 344,841 rows |

## 2. What we built, in the order it paid

| step | board |
|---|---:|
| Pooled LightGBM baseline | 405.75 |
| Per-airport and per-operator tree models | 386.22 |
| + neural network (MLP) blended per airport | 324.28 |
| MLP seed averaging, 1 → 18 seeds | 295.21 |
| + TabNet override at LIRF | 293.67 |
| + FT-Transformer as a third member | 289.78 |
| FT-Transformer seed averaging and three architectures (small, large, deep) | 286.53 |
| ADS-B (adsb.lol) corrector on top, 3 → 7 airports, more training months (v46/v48) | 272.17 |
| + winter morning de-icing heuristic on uncovered frost flights (v49) | 271.94 |
| **+ July convective thunderstorm offset on uncovered TS flights (v50)** | **271.87** |


Supporting infrastructure: time-ordered validation splits with a separate weight-fitting
split; pre-registered pass bars for every experiment; a convergence audit; a board-score log;
a reproducibility check that rebuilds uploads row by row; an end-to-end prediction script.

## 3. What we explored

✅ = paid and is in the final model · ❌ = tested and closed · ➖ = real but too small to use

### 3.1 Model families and ensembling

| idea | outcome | evidence |
|---|---|---|
| LightGBM pooled / per-airport / per-operator / stacked | ✅ trees (per-airport + per-operator blend) | `PIPELINE_REVIEW.md` |
| XGBoost, ridge, naive off-block baselines | ❌ weaker than LightGBM | `PIPELINE_REVIEW.md` |
| CatBoost as a third member | ❌ board v26 295.48, no gain | `TAXI_OUT_DRIVERS.md` §2m |
| MLP with entity embeddings | ✅ the largest single gain (−62) | plans 02–06 |
| MLP seed averaging (1 → 18) | ✅ −29 on the board | `tests_lirf_seeds`, board log |
| MLP variants: wider, deeper, residual, regularized, big-batch | ❌ fail at equal member count | `tests_hp_diversity` |
| MLP backbones: PLE numeric embeddings, TabM, TabM + PLE | ❌ all worse | `tests_backbone` |
| Per-airport specialist MLPs | ❌ broke at LIRF (tail scarcity) | §2y, `pipeline_per_airport_mlp` |
| TabNet | ✅ only at LIRF (override, −1.5) | `tests_tabnet` |
| TabNet seed averaging 5 → 11 | ❌ board +0.91 | `tests_tabnet/PROBE_v33.md` |
| FT-Transformer (small) | ✅ −3.9 | `tests_ftt` |
| FT-Transformer seed averaging, large and deep variants | ✅ −2.9 in total | `tests_ftt_large`, `tests_ftt_deep` |
| FT-Transformer at LIRF (plain or down-only guard) | ❌ weight solves to 0 | `tests_ftt_lirf_guard` |
| Transductive pretraining for FT-Transformer | ❌ +19.8 s worse | `tests_ftt_pretrain` |
| Sequence / neighbour models (GRU state, queue attention, TabR retrieval) | ❌ same errors as the MLP (corr 0.98) | `tests_seqrnn`, `tests_retrieval_screen`, §2k |
| Matched / unmatched router | ❌ lost to the blend | `tests_matched_unmatched_router` |
| Learned stacker, stage-2 residual boosting, month-fold residual corrector (forum idea) | ❌ overfit / Jan + Jul +0.37 | §2v, plan 04, `tests_corrector` |
| Tree-side seed averaging | ❌ Jan + Jul +0.16 | `tests_tree_seeds` |
| Finer blend-weight bucketing (by WTC, runway, cluster) | ❌ over-tuned, quarantined | `overtuned/README.md` |

### 3.2 Target, loss and outliers

| idea | outcome | evidence |
|---|---|---|
| Keep every row including the tail (`keep_all`) | ✅ board-confirmed (winsorizing: 496) | §2ai, `config.py` |
| Winsorize / drop the tail, drop date-artifact rows | ❌ | §2g, §2l, §2ai |
| log1p, Huber, Poisson, Gamma/Tweedie, quantile ladder | ❌ all de-weight the tail | `tests_log_target`, §2i, §2ac, `tests_quantile_ladder` |
| Residual target (y − off-block delta), floor + delta split, hurdle model | ❌ | §2x, §2j, plan 07 |
| Prediction cap at 7,200 s / floor at 240 s | ❌ cap board 491; floor worse | plan 07, `tests_pred_floor` |

### 3.3 Features from the provided data

| idea | outcome | evidence |
|---|---|---|
| Actual off-block time (AOBT) features | ✅ core feature set (`with_aobt`) | `PLAN.md` |
| Congestion counts (departures/arrivals in time windows, 5/10/30 min) | ➖ tiny; kept in some arms | `tests_congestion_feature`, §2p |
| Neighbour wake-turbulence sequencing | ➖ real but small | `tests_wtc_sequencing`, §2ag |
| Specific flight-number identity, as ID or target statistic | ❌ does not survive the blend | §2t, §2u |
| Own-flight arrival times, LOBT − schedule | ❌ near zero | §2s, §2ak |
| Runway configuration, active arrival runway | ❌ | `tests_runway_config`, §2z |
| Stand-turnover consistency, label correction | ❌ (the premise was retracted) | `tests_stand_turnover`, §2ae |
| Unsupervised features (stand/runway embeddings, day regimes) | ➖ 1 % of gain | `RETROSPECTIVE.md` |
| Recency weighting (trees), season-matched weighting (MLP) | ❌ board v37 +2.07 | `pipeline_lightgbm_recency`, `tests_season_weight` |
| Feature drift, 2025 vs 2026 | ✅ checked: none | `tests_drift_audit`, §2ad |

### 3.4 External (open) data

| source | outcome | evidence |
|---|---|---|
| METAR weather: generic multi-feature residual models | ❌ within noise / overfit | `tests_weather_metar`, §2r, §2w |
| METAR frost flag: targeted de-icing offset (+30s) on uncovered January flights | ✅ **−0.23 on board (v49 = 271.94)** | `tests_weather_metar/RESULT_v49.md` |
| METAR thunderstorm offset (+200 s) on uncovered July flights | ✅ v50 = 271.87 (−0.07; expected −0.30 to −0.50) | `tests_weather_metar/RESULT_v50.md` |
| Weather offsets checked out of sample (fit on 6 months, score the 7th) | v50's offsets mildly supported (−0.07); fitted per-condition offsets −0.16 but worse in Jan; **per-airport offsets (v51–v54) not supported**, never uploaded | `tests_weather_validated/RESULT.md` |
| Feb–Jun as extra ADS-B corrector training (new held-out base) | ❌ +0.07 pooled, 2/7 months: residuals from a differently-built base | `tests_febjun/RESULT.md` |
| ADS-B corrector at LIRF, flights with an NM record only (Jan 2026) | ✅ LIRF −4.2 locally (5/6 months); **board v55 = 271.764 (−0.11)** | `tests_lirf_adsb/RESULT.md` |
| ADS-B corrector at LFPG (track + record rows) | ✅ −0.098 locally (6/7); **board v56 = 271.5625 (−0.20)** | `tests_round9/RESULT.md` |
| 7-airport corrector on record rows only; EGLL own model | ❌ +0.26 (0/7); −0.005 not robust | `tests_round9/RESULT.md` |
| OpenStreetMap stand-to-runway straight-line distance | ➖ real, negligible after the stand×runway ID | §2h |
| Public holiday calendars | ❌ | `tests_holiday`, §2ab |
| EUROCONTROL ATFM delay statistics | ❌ negative on the ceiling | `tests_atfm_delay`, §2ah |
| ADS-B (adsb.lol) ground tracks → residual corrector at 7 airports | ✅ **−14.3 on the board** (v43 281.86, v44 279.15, v45 273.60, v46 272.26). Each step pre-registered and confirmed on a fresh 2025 season first | `tests_adsb_full/RESULT.md` |
| ADS-B corrector at LIRF | ❌ Aug 2025 worse (654 → 675) | `tests_adsb_full/PREREGISTERED_v46.md` |
| ADS-B features inside the tree model | ❌ tree −7 alone, +0.08 once corrected | `tests_adsb_full/PREREGISTERED_tree.md` |
| Tabular columns inside the ADS-B corrector | ❌ 7/7 months locally, **board v47 +0.47** (year shift) | `tests_adsb_full/PREREGISTERED_interactions.md` |
| Traffic features from other aircraft's ADS-B | ❌ drift gate: matched-traffic counts track receiver coverage (standardized shift up to 0.48) | `tests_adsb_full/PREREGISTERED_v48.md` |
| Corrector seed averaging (5 seeds) | ✅ v48 = 272.17 (−0.09) | `tests_adsb_full/PREREGISTERED_v48.md` |
| Looser ADS-B matching at EDDF/LEMD | ❌ failures have zero ground points | `tests_runway_queue/RESULT.md` |
| Runway queue / runway mode features (records) | ❌ stable across years but pooled +2.11, LIRF +10 | `tests_runway_queue/RESULT.md` |
| Routed taxi distance (apt.dat) | ❌ ceiling −0.14 s (unseen stand-runway pairs only) | `tests_runway_queue/routed_ceiling.txt` |

### 3.5 The tail (the few very long taxi-outs)

| idea | outcome | evidence |
|---|---|---|
| Diagnosis: where the squared error sits | done: ~44 % in ~150 rows, mostly LIRF without a flight record | `tests_tail_diagnosis` |
| 2025 structure: true value = scheduled delay, or ≈ one day | found, but did **not** transfer to 2026 | `tests_tail_study` |
| LIRF identity override built on that structure | ❌ board +0.86, twice | `tests_lirf_identity` |
| Carrier-level echo hedge for LIRF no-record rows (forum idea) | ❌ carrier predicts the echo, but the hedge is far worse than the model (Jan + Jul +60) | `tests_carrier_echo` |

### 3.6 Training and validation mechanics

| idea | outcome | evidence |
|---|---|---|
| Convergence audit (runs stopped at their cap) | fixed; 3 early verdicts corrected | `CONVERGENCE_AUDIT.md` |
| Refit epoch count, longer training for ensembles | ❌ flat on Jan + Jul | `tests_refit_epochs` |
| Snapshot averaging | ❌ | `tests_snapshot_avg` |
| Weight-fitting surface (Jul–Aug vs closer windows vs flat) | ✅ fitted beats flat; the surface barely matters | §2n, `tests_seasonal_flat_weights` |
| Exact vs early-stopped weight solver | the early stop acts as useful shrinkage; kept | solver docstring |
| TabNet weight at its exact board optimum (0.459) | ✅ −0.065, predicted to 0.0001 s | v35 |

## 4. What we did not explore, and why

| idea | why not |
|---|---|
| **Probing the public board** (learning individual rows' truths from score differences) | Deliberately excluded: it fits the test set, not a model. The moderator hinted at a hidden final stage, where it would not transfer. |
| **Whether post-pushback ADS-B is allowed** | Used (§3.4) but never confirmed by the organizers. A forum question raised it as possible leakage; no explicit answer. v42 is kept as the clean final if the answer is no. |
| **ADS-B at LFPG and LTFM** | No ground coverage in adsb.lol; the OpenSky database (which might have it) is ruled out by the organizers. |
| **Taxi-route geometry from a taxiway graph** (e.g. X-Plane `apt.dat`, which other teams use) | We only tested straight-line OSM distance, which added nothing beyond the stand×runway ID. The actual routed path length, and LSZH (OSM failed there), were not built. |
| **Graph neural networks over the airport surface** | Not attempted; would need the taxiway graph above. |
| **A departure-queue simulation** (runway sequencing, holding) | Only captured through features and a queue-attention model; no explicit simulator. |
| **Other deep-tabular families** (NODE, SAINT, T2G-Former, ExcelFormer) | Not tried. NODE was expected to behave like the trees, and SAINT's row attention is the neighbour approach that closed. |
| **More FT-Transformer variants** (deep and wide, other learning rates) and a tuned FTT/TabNet search | Only three FTT shapes tried; the variety gains were shrinking (−0.64, −0.25, −0.23). The deep 3 → 6 seed run is in progress. |
| **Explicitly modelling the "one-day" label artifact for 2026** | Studied, not shipped: the 2025 rule regressed on the 2026 board, and the 2026 behaviour can't be seen locally. |
| **Data beyond 2025** (earlier years, network-manager B2B flow data) | Not provided; licensing unclear. |
| **Why the leaders are still ~50 s ahead** | Partly ADS-B (it closed 14 s of it). The rest is not identified: not the tail alone, not weather or stand geometry. |

## 5. Lessons worth keeping

0. **Don't close an idea on a weak test.** ADS-B was first "closed" on 3 days and one crude feature; tested properly, it became the biggest gain since the neural models.

1. **On this metric the tail decides the model's shape.** Keep every row and train on plain squared error; every tail-shrinking trick lost on the board.
2. **Averaging and variety paid more than cleverness.** MLP seeds (−29), then different neural families and FT-Transformer shapes (−8.3); inside the MLP family nothing paid.
3. **Validate on the board's months.** Six ideas passed on Sep–Dec and failed on Jan + Jul, and the one tested on the board lost.
4. **Local checks under-read averaging and variety.** For those questions a pre-registered board probe with a fallback was the only honest instrument.
5. **Write the pass bar before the result, and treat a capped run as unresolved.** Both prevented false closures and false passes.
