# `exploration/` — the experiments behind the solution

The submitted solution is in [`../solution/`](../solution/); its method is described in
[`../solution/WRITEUP.md`](../solution/WRITEUP.md). This folder holds everything that was tried on the
way: 58 experiments, the candidate models that were not selected, the early analysis scripts and the
decision history.

**Start with [`EXPLORATION_SUMMARY.md`](EXPLORATION_SUMMARY.md):** one page listing everything tried,
with its outcome.

**About the scripts.** They were written when this folder and `solution/` were one folder. A small
helper (`../tools/repo_layout.py`) lets a script here find the shared code, models and data in
`../solution/`. The scripts were updated mechanically for the split; a sample was re-run and
reproduced its earlier numbers, but not all of them were re-run.

Eight experiment folders stayed in `../solution/`, because the final pipeline reads them:
`tests_adsb_full`, `tests_lirf_adsb`, `tests_round9`, `tests_ftt`, `tests_ftt_wide`, `tests_tabnet`,
`tests_final`, `tests_repro`.

## 1. Experiments (`tests_*/`)

Each folder is one question. Most have a `PREREGISTERED.md` (the pass bar, written first) and a
`RESULT.md`; a few early ones have only a script and its output.

**Base-model members and ensembling**

| folders | question |
|---|---|
| `tests_ftt_large/`, `tests_ftt_deep/` | the large and deep FT-Transformer shapes |
| `tests_ftt_lirf_guard/`, `tests_ftt_pretrain/` | FT-Transformer at LIRF; transductive pretraining |
| `tests_backbone/`, `tests_hp_diversity/`, `tests_arch_diversity/` | other MLP backbones and variants |
| `tests_seqrnn/`, `tests_retrieval_screen/` | sequence and neighbour models |
| `tests_tree_seeds/`, `tests_lirf_seeds/`, `tests_seed_robustness/`, `tests_snapshot_avg/` | seed and snapshot averaging |
| `tests_refit_epochs/`, `tests_mlp_convergence/` | epoch counts for the full refit; convergence |
| `tests_season_weight/`, `tests_seasonal_flat_weights/` | season weighting; fitted against flat blend weights |
| `tests_corrector/`, `tests_matched_unmatched_router/`, `tests_matched_unmatched_split/` | early residual correctors and routers |

**Target and loss**

| folders | question |
|---|---|
| `tests_log_target/`, `tests_poisson_objective/`, `tests_quantile_ladder/` | other objectives |
| `tests_pred_floor/`, `tests_delta_target/`, `tests_label_correction/` | prediction floor; residual target; label correction |

**Features from the provided data**

| folders | question |
|---|---|
| `tests_congestion_feature/`, `tests_wtc_sequencing/` | congestion windows; wake-category sequencing |
| `tests_runway_config/`, `tests_runway_queue/`, `tests_wind_runway/` | runway configuration, queue and wind |
| `tests_stand_turnover/`, `tests_lobt_sched/`, `tests_composition_correction/`, `tests_telemetry_identity/` | stand turnover; LOBT against schedule; other record features |
| `tests_holiday/`, `tests_atfm_delay/` | holiday calendars; EUROCONTROL ATFM delay statistics |
| `tests_drift_audit/` | feature drift, 2025 against 2026 |

**The tail and LIRF**

| folders | question |
|---|---|
| `tests_tail_diagnosis/`, `tests_tail_study/` | where the squared error sits |
| `tests_lirf_identity/`, `tests_lirf_flightpath/`, `tests_lirf_proxy_probe/`, `tests_carrier_echo/` | LIRF no-record rows: identity override and related ideas |
| `tests_copymix/`, `tests_lirf_spec/` | ideas from other teams: copy mixture; LIRF no-record specialist |

**ADS-B**

| folders | question |
|---|---|
| `tests_adsb_coverage/` | first coverage check |
| `tests_febjun/` | the Feb–Jun 2025 held-out base; more corrector training months |
| `tests_round10/` – `tests_round13/` | corrector size; the training base |
| `tests_ctx/` | a context corrector on every row |

**Weather**

| folders | question |
|---|---|
| `tests_weather_metar/`, `tests_weather_validated/` | METAR features and offsets; their out-of-sample check |

**Checks and audits**

| folders | question |
|---|---|
| `tests_newdata/` | the pipeline on new data |
| `tests_next/`, `tests_local_vs_board/` | audit of early closures; local scores against board scores |

## 2. Candidate base models (`pipeline_*/`, `overtuned/`)

Each `pipeline_*/` folder is one candidate model with its `train.py`. The eight folders the final model
uses are in `../solution/`; the 26 here were tried and not selected (`PIPELINE_REVIEW.md` reviews the
early ones). `overtuned/` holds candidates that were quarantined as over-tuned.

## 3. Documents

| path | what it is |
|---|---|
| [`EXPLORATION_SUMMARY.md`](EXPLORATION_SUMMARY.md) | one page: everything tried, with its outcome (current) |
| `PIPELINE_REVIEW.md`, `TAXI_OUT_DRIVERS.md`, `RETROSPECTIVE.md` | review of the early candidate pipelines; the full catalogue of features and drivers tried; a retrospective. Written mid-build, not maintained |
| `CONVERGENCE_AUDIT.md`, `DATA_QUALITY_AUDIT.md` | audits of training convergence and of the data |
| [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) | the first look at the data: taxi-time distribution, odd values, missing data, months, airports, runways, aircraft types |
| `PLAN.md`, `plans/`, `EXTERNAL_PLAN_CROSSCHECK*.md`, `result.md` | plans and decision history |

Paths inside these older documents are written as they were when everything was one folder.

## 4. Early analysis scripts and their outputs

Kept as the record of the first weeks; not part of the final pipeline.

| path | what it is |
|---|---|
| `stage1_audit.py` … `stage17_iqr_outlier_profile.py` | the staged audit: leakage, label quality, split, controls, ablations, tuning |
| `scratch_*.py` | exploratory scripts |
| `compare.py`, `run_all.py`, `run_all.ipynb`, `run_queue.py`, `harvest_results.py`, `build_*.py` | runners and result collectors for the candidate pipelines |
| `compare_summary__*.json`, `results.csv`, `pipeline_*.csv`, `logs/`, `graph_view/` | their outputs |
