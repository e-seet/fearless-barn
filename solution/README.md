# `solution/` — the final solution

**Read [`WRITEUP.md`](WRITEUP.md) first.** It describes the model, the decisions, the results and how
to reproduce them. This page is a map of the folder.

This folder holds the final pipeline, the code that trains its models, and the saved models and files
it reads. Every other experiment, and the candidate models that were not selected, are in
[`../exploration/`](../exploration/).

## Run it

```bash
pip install -r requirements.txt
python final_pipeline.py --movements <movements.parquet> --points-dir <adsb_dir> --fetch --out <pred.parquet>
```

## The pipeline

| path | what it is |
|---|---|
| [`final_pipeline.py`](final_pipeline.py) | the whole recipe, stages 1–6, for any movements file |
| [`predict_new.py`](predict_new.py) | stage 1 (the base model) from saved models |
| [`submit.py`](submit.py) | assembles base-model predictions for the ranking set; validates a submission file |
| [`config.py`](config.py), [`common/`](common/) | paths, data loading, feature building, splits |
| [`requirements.txt`](requirements.txt) | Python dependencies |
| [`submissions/`](submissions/) | the final round file, key board files, `board_scores.csv` (every score with its pre-written expectation) |

## Where each stage lives

The `tests_*` folders kept here hold a stage's code and saved model, together with the experiments
that selected it.

| stage | code | saved model or weights |
|---|---|---|
| 1. base model | `pipeline_per_airport/`, `pipeline_per_operator/` and `pipeline_airport_operator_blend/` (trees and their blend; `pipeline_per_airport_congest/` is imported by their training code), `pipeline_mlp_antigravity/` (MLP), `pipeline_ftt_multimodel/` and `tests_ftt/` (FT-Transformers), `pipeline_catboost_antigravity/` (CatBoost), `pipeline_tabnet_lirf_override/` and `tests_tabnet/` (TabNet at LIRF) | `tests_ftt_wide/final_weights_v59_s4.json`; saved outputs `tests_ftt_wide/v59_base.parquet` (ranking set) and `tests_final/final_base.parquet` (final set) |
| 2. ADS-B corrector | `tests_adsb_full/` (`fetch.py`, `features.py`) | `tests_adsb_full/model_v61/` (earlier: `model_v48`, `model_v57`, `model_v60`) |
| 3. weather offsets | in `final_pipeline.py` | two constants |
| 4. LIRF corrector | `tests_lirf_adsb/` | `tests_lirf_adsb/model_v55/` |
| 5. LFPG corrector | `tests_round9/` | `tests_round9/model_v56/` |
| 6. LOBT window | in `final_pipeline.py` | a rule |

## Other folders

| path | what it is |
|---|---|
| `tests_final/` | the final round: the re-check on February–June, the ADS-B fetch and re-basing, the build |
| `tests_repro/` | sha256 manifests of the files git does not hold, and rebuild checks |
| `artifacts/` | metadata of every trained run (parameters, metrics, split definitions) |

The trained stage-1 models (about 37 GB) are not in git; see write-up §8.

`submit.py` and `common/` can also assemble the candidate models that were not selected. Their code is in
`../exploration/pipeline_*/`, so that only works with `../exploration` on the Python path.
