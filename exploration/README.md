# `exploration/` — the solution and every experiment behind it

**Read [`WRITEUP.md`](WRITEUP.md) first.** It describes the model, the decisions, the results and how
to reproduce them. This page is only a map of the folder.

The folder is flat: it grew one experiment at a time, and the pipeline reads several of these paths
directly, so the layout was left as it is. The groups below say what each part is for.

## 1. The final solution

| path | what it is |
|---|---|
| [`final_pipeline.py`](final_pipeline.py) | the whole recipe, stages 1–6, for any movements file |
| [`predict_new.py`](predict_new.py) | stage 1 (the base model) from saved models |
| [`submit.py`](submit.py) | assembles base-model predictions for the ranking set; validates a submission file |
| [`config.py`](config.py), [`common/`](common/) | paths, data loading, feature building, splits |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`submissions/`](submissions/) | the final round file, key board files, `board_scores.csv` (every score with its pre-written expectation) |

**Saved models and weights the pipeline reads:**

| stage | path |
|---|---|
| 1. blend weights | `tests_ftt_wide/final_weights_v59_s4.json` |
| 1. saved outputs | `tests_ftt_wide/v59_base.parquet` (ranking set), `tests_final/final_base.parquet` (final set) |
| 2. ADS-B corrector | `tests_adsb_full/model_v61/` (earlier versions: `model_v48`, `model_v57`, `model_v60`) |
| 4. LIRF corrector | `tests_lirf_adsb/model_v55/` |
| 5. LFPG corrector | `tests_round9/model_v56/` |

The trained stage-1 models (about 37 GB) are not in git; see write-up §8.

## 2. Documents

| path | what it is |
|---|---|
| [`WRITEUP.md`](WRITEUP.md) | **the method (current)** |
| [`EXPLORATION_SUMMARY.md`](EXPLORATION_SUMMARY.md) | one page: everything tried, with its outcome (current) |
| `PIPELINE_REVIEW.md`, `TAXI_OUT_DRIVERS.md`, `RETROSPECTIVE.md` | review of the early candidate pipelines; the full catalogue of features and drivers tried; a retrospective. Written mid-build, not maintained |
| `CONVERGENCE_AUDIT.md`, `DATA_QUALITY_AUDIT.md` | audits of training convergence and of the data |
| `PLAN.md`, `plans/`, `EXTERNAL_PLAN_CROSSCHECK*.md`, `result.md` | plans and decision history, including two mid-build snapshots of this README and of the submission notes |

## 3. Base-model training code (`pipeline_*/`)

Each `pipeline_*/` folder is one candidate model with its `train.py`. Five are members of the shipped
base model:

| folder | member |
|---|---|
| `pipeline_airport_operator_blend/` | trees: per-airport + per-operator LightGBM |
| `pipeline_mlp_antigravity/` | the entity-embedding MLP |
| `pipeline_ftt_multimodel/` (with `tests_ftt/run_ftt.py`) | FT-Transformers and the per-airport blend |
| `pipeline_catboost_antigravity/` | CatBoost |
| `pipeline_tabnet_lirf_override/` (with `tests_tabnet/`) | TabNet at LIRF |

The other `pipeline_*/` folders are candidates that were not selected; `PIPELINE_REVIEW.md` reviews
them. `overtuned/` holds candidates that were quarantined as over-tuned.

`artifacts/` holds the metadata of every trained run (parameters, metrics, split definitions). The
model files themselves are not in git.

## 4. Experiments (`tests_*/`)

Each folder is one question. Most have a `PREREGISTERED.md` (the pass bar, written first) and a
`RESULT.md`; a few early ones have only a script and its output.

**Base-model members and ensembling**

| folders | question |
|---|---|
| `tests_ftt/`, `tests_ftt_large/`, `tests_ftt_deep/`, `tests_ftt_wide/` | the four FT-Transformer shapes; the v59 base |
| `tests_ftt_lirf_guard/`, `tests_ftt_pretrain/` | FT-Transformer at LIRF; transductive pretraining |
| `tests_tabnet/` | TabNet at LIRF |
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

**ADS-B (stages 2, 4, 5)**

| folders | question |
|---|---|
| `tests_adsb_coverage/` | first coverage check |
| `tests_adsb_full/` | fetch, matching, features, the stage-2 corrector (rounds 1–7) |
| `tests_febjun/` | Feb–Jun 2025 held-out base; more corrector training months |
| `tests_lirf_adsb/`, `tests_round9/` | the LIRF and LFPG correctors |
| `tests_round10/` – `tests_round13/` | corrector size; the training base |
| `tests_ctx/` | a context corrector on every row |

**Weather (stage 3)**

| folders | question |
|---|---|
| `tests_weather_metar/`, `tests_weather_validated/` | METAR features and offsets; their out-of-sample check |

**The final round, reproducibility and audits**

| folders | question |
|---|---|
| `tests_final/` | the final round: re-check on Feb–Jun, ADS-B fetch and re-basing, the build |
| `tests_newdata/`, `tests_repro/` | the pipeline on new data; sha256 manifests and rebuild checks |
| `tests_next/`, `tests_local_vs_board/` | audit of early closures; local scores against board scores |

## 5. Early analysis scripts and their outputs

Not part of the final pipeline. Kept as the record of the first weeks.

| path | what it is |
|---|---|
| `stage1_audit.py` … `stage17_iqr_outlier_profile.py` | the staged audit: leakage, label quality, split, controls, ablations, tuning |
| `scratch_*.py` | exploratory scripts |
| `compare.py`, `run_all.py`, `run_all.ipynb`, `run_queue.py`, `harvest_results.py`, `build_*.py` | runners and result collectors for the candidate pipelines |
| `compare_summary__*.json`, `results.csv`, `pipeline_*.csv`, `logs/`, `graph_view/` | their outputs |
