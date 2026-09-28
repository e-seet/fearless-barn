# fearless-barn — PRC Data Challenge 2026, taxi-out time: method write-up

**Status (2026-09-29):** best board score **v41 = 286.8032**, also the last upload (ranking
uses the last submission). v41's FTT slot = 6 small (d64×3) + 9 large (d192×3) + 3 deep (d96×6) FT-Transformers, one third per family. **Ranking uses the last submission**, so the final upload must be the chosen
file. v35 = v32 (6-seed FT-Transformer) with TabNet's LIRF weight at its exact
board optimum, 0.459.
This file is the current summary. `README.md` and `SUBMISSION.md` are
mid-build snapshots and point here.

---

## 1. Task and metric

Predict taxi-out time (seconds from off-block to takeoff) for every departure in
the scored set: January and July 2026 at ten European airports (EDDF, EDDM, EGLL,
EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM). Training data: all of 2025,
2,085,047 departures. The metric is **pooled RMSE**.

The metric shapes everything. About 0.02 % of rows (taxi-outs above 7,200 s)
carry 15–38 % of the squared error, depending on the period. Under L2 the
optimal prediction is the conditional *mean*, so anything that shrinks the tail
(winsorizing, robust losses, prediction caps) loses (§5).

## 2. The final model

A per-airport blend of three model families, refit on all twelve 2025 months,
plus an airport-specific override at Rome:

```
                   ┌─ tree family: per-airport + per-operator LightGBM (airport_operator_blend)
per-airport blend ─┼─ MLP: entity-embedding MLP (256,128,64), mean of 18 seeds
 (weights per      └─ FT-Transformer (column attention), mean of 3 seeds (v31) / 6 (v32)
  airport)
        │
        └─ at LIRF only: 0.45 × blend + 0.55 × TabNet (mean of 5 seeds)
```

- **Per-airport weights** are solved on `forward_inner`: train Jan–Jun, holdout
  Jul–Aug, a surface no scored model is fitted on. They are then frozen and
  applied to the scored set. FTT's weight at LIRF solves to exactly 0 in every
  solve (six independent solves).
- **The TabNet override** replaces nothing outside LIRF. The two neural additions
  act on disjoint rows, so their board effects add exactly: v31 landed **0.0014 s**
  from the score predicted from v29 and v30 before upload.
- **Training target policy:** `keep_all`. Every valid row is kept, the tail
  included. Predictions are clipped only to [60, 86,400] s.
- **Refit:** every member is retrained on all of 2025 with hyperparameters and
  epoch counts frozen from the time-ordered `forward` split. `full` has no
  holdout, so nothing is selected on it.

Registered arms: `ftt_multimodel` (tag `ftt3_mlp18`, and `ftt6_mlp18` for v32)
and `tabnet_lirf_override` (tag `lirf_w0.55_ftt3`, and `lirf_w0.55_ftt6` for v32).
`submit.py --arm tabnet_lirf_override --tag <tag> ...` rebuilds either.

## 3. Board history: what moved the score

| version | change | board | Δ |
|---|---|---:|---:|
| v2 | pooled LightGBM | 405.75 | |
| v3 | per-airport stack (trees only) | 386.22 | −19.5 |
| v13 | + 1-seed MLP in a per-airport blend | 324.28 | −61.9 |
| v14 | 3 MLP seeds | 306.32 | −18.0 |
| v20 | 6 MLP seeds | 299.47 | −6.9 |
| v22 | blend weights solved on clean (non-stale) MLPs | 298.07 | −1.4 |
| v24 | 12 MLP seeds | 295.75 | −2.3 |
| v25 | 18 MLP seeds | 295.21 | −0.5 |
| v29 | + TabNet override at LIRF (w = 0.55) | 293.67 | −1.5 |
| v30 | + FT-Transformer as a third blend member (off LIRF) | 291.34 | −3.9 (vs v25) |
| v31 | v30 + v29's TabNet override | 289.78 | −5.4 (vs v25) |
| **v32** | **FT-Transformer 3 → 6 seeds (LIRF unchanged)** | **287.99** | **−1.79** |
| v33 | probe: TabNet 5 → 11 seeds at LIRF | 288.90 | +0.91 (closed) |
| v34 | probe: FTT 6 → 7 seeds | 288.03 | +0.04 (tie) |
| v35 | v32 + TabNet LIRF weight 0.459 (board-optimal) | 287.9247 | −0.065 (predicted 287.9248) |
| v36 | FTT slot = 6 small + 3 large (d192) FT-Transformers | 287.2876 | −0.637 |
| v37 | probe: season-weighted MLPs | 289.3555 | +2.07 (closed) |
| v39 | FTT slot = 6 small + 6 large, large share 1/3 | 287.0370 | −0.251 |
| **v41** | **FTT slot = 6 small + 9 large + 3 deep (d96×6), 1/3 each** | **286.8032** | **-0.234** |

Two board results that settled the tail question (§5): **v18** (v14 with
predictions capped at 7,200 s) = **491.35**; **v27** (training target winsorized
at 7,200 s) = **496.09**.

**The three big levers, in order:** blending a neural model with the trees
(−62), averaging MLP seeds (−28 over 1 → 18 seeds), and adding architecturally
different members that are *both* decorrelated *and* competitive standalone
(TabNet at LIRF, FTT elsewhere: −5.4), then averaging FTT seeds (−1.8).

**The board pays more for seed averaging than the local surfaces predict.**
MLP 1 → 3 seeds: −0.9 local, −18 board. FTT 3 → 6: −0.13 to −0.30 local
in-band, −1.79 board. Local checks are underpowered for seed-count questions.

## 4. Validation protocol

| split | train | holdout | used for |
|---|---|---|---|
| `forward` | Jan–Aug | val Sep–Oct, test Nov–Dec | primary honest split: early stopping, model comparison |
| `forward_inner` | Jan–Jun | Jul–Aug | solving blend weights (out-of-sample) |
| `seasonal_full` | Feb–Jun, Aug–Nov | val Dec, test **Jan + Jul** | the board's months; confirmation before any build |
| `full` | all 12 months | none | the submission refit only; selection forbidden |

Rules that paid for themselves:

- **Pre-register the bar before the number exists** (`tests_*/PREREGISTERED*.md`).
- **Multi-seed, paired by seed.** MLP single-seed val SD is ~3.5 s, so a
  single-seed delta below that is uninterpretable. Compare the same seed with
  and without the change.
- **A forward-test (Nov–Dec) pass needs a seasonal_full (Jan + Jul) replication.**
  On 2026-09-23 two leads passed forward and failed there: an ensemble-epoch
  effect and a deeper MLP variant. FTT, which transferred to the board, had
  been confirmed on seasonal_full first.
- **Treat a run that hit its epoch or round cap as unresolved, not failed.**
  Three verdicts in this build were first read off truncated runs; FTT was
  nearly closed at cap 60 and later became v30 (`CONVERGENCE_AUDIT.md`).
- **Local → board mapping:** board ≈ −246 + 2.24 × local val, residual SD ~9
  board-seconds (`tests_local_vs_board/RESULT.md`). One submission cannot
  resolve a local effect under ~4 s. Composition checks on disjoint rows, as
  with v31, are the exception: they predict exactly.

## 5. What did not work (closed, with the evidence)

| idea | result | where |
|---|---|---|
| winsorize / drop the tail in training | `keep_all` wins; board v27 = 496 | `config.py` OUTLIER_POLICIES, `artifacts/clip/` |
| cap predictions | board v18 = 491 | plan 07 |
| log1p target, Huber, Poisson, quantile ladder | all lose; they de-weight the tail | `tests_log_target`, `TAXI_OUT_DRIVERS.md` §2i, `tests_poisson_objective`, `tests_quantile_ladder` |
| hurdle model (P(tail) × magnitude) | lost twice; 0.02 % base rate | plan 07, `delay_twostage_antigravity` |
| LIRF "identity" override (no flight record → scheduled delay) | right on its rows locally, **+0.86 on the board twice** (v15, v17) | `tests_lirf_identity`, `tests_tail_diagnosis` |
| neighbour / sequence models (GRU, queue attention, retrieval) | residual corr 0.98 with the MLP | `tests_seqrnn`, `tests_arch_diversity`, `tests_retrieval_screen` |
| hyperparameter tuning | 6.5 s for a full sweep vs 27 s for one reasoned constant | `PIPELINE_REVIEW.md` |
| MLP hyperparameter diversity (wide / deep / reg / big-batch) | fails at equal member count; `deep` passed forward, failed Jan + Jul | `tests_hp_diversity` |
| MLP backbone: PLE numeric embeddings, TabM (k = 32), TabM+PLE | all fail forward; PLE +17.8 s, TabM +8.8 s val, TabM+PLE +11.9 s at LIRF | `tests_backbone` |
| larger FT-Transformer (d_token 192), alone or mixed with the small one | alone worse (Jan + Jul 266.8 vs 257.2); mixed: forward −0.3/−0.6, Jan + Jul +0.03 | `tests_ftt_large` |
| transductive masked-feature pretraining for FTT | +19.8 s worse on the same seed | `tests_ftt_pretrain` |
| more epochs for the ensemble / changing the refit epoch count | flat on Jan + Jul, +19 s pooled | `tests_refit_epochs` |
| tree-side seed averaging (1 → 6 LightGBM seeds) | forward −0.09 / −0.18, **Jan + Jul +0.16** | `tests_tree_seeds` |
| FTT at LIRF, downward-only guard | weight solves to 0 | `tests_ftt_lirf_guard` |
| FTT 6 → 7 / 9 seeds | Jan + Jul +0.06 (9); board probe v34 (7) tied, 288.03 | `tests_ftt/PROBE_v34.md` |
| extra TabNet seeds | local −0.5 s (closed); board probe v33 (11 seeds) **+0.91 worse** | `tests_tabnet/RESULT.md`, `tests_tabnet/PROBE_v33.md` |

A convergence audit (`CONVERGENCE_AUDIT.md` §9) re-checked every closed
outlier verdict for truncated runs; all stand.

## 6. Where the remaining error is

`tests_tail_diagnosis/RESULT.md`, on the v31 analogue for Jan + Jul 2025:

- **55.9 %** of squared error is in-band (≤ 7,200 s), the only mass that
  model quality and averaging have ever moved;
- **28.2 % is in 7 "date-artifact" rows** (≈ one day). One LFPG row alone is
  17.8 %, and its features show no sign of the delay;
- the rest is a LIRF no-flight-record tail whose only feature-based rule is
  the board-rejected identity override.

## 7. Reproducibility

`tests_repro/rebuild_check.py` rebuilds submissions through `submit.py`'s own
code path (provenance gate → `predict_ranking` → `np.rint`) and compares every row
with the uploaded file. It also records, through a Python audit hook, every file
the rebuild opens.

| submission | rebuilt vs uploaded |
|---|---|
| **v31** | identical on all 344,841 rows |
| v30 | identical on all 344,841 rows |
| v29 | identical except 13,525 LIRF rows, each off by exactly 1 s (the upload was built by a script that truncated; `submit.py` rounds) |

The rebuild opened 60 files. **52 are not in git (0.43 GB):** trained runs under
`artifacts/runs/` (236 MB), the feature cache (130 MB), TabNet predictions in
`tests_tabnet/build/` (37 MB), FTT `full` predictions (12 MB; 25 MB with v32's
seeds 45–47), the split file, and `submitting.parquet`. Path, size and sha256 of
each are in `tests_repro/dependency_manifest.csv`. **Those files are the backup
set.** The feature cache and runs can also be regenerated from
`prc-2026-datasets/` with the committed code, at the cost of retraining.

---

Detailed records: `plans/` (numbered decision history, latest plan 14),
`PIPELINE_REVIEW.md` (arm tiers), `TAXI_OUT_DRIVERS.md` (feature
investigations), `tests_*/RESULT.md` (one per experiment),
`submissions/board_scores.csv` (every board score).

### Predicting new data (`predict_new.py`)

A hidden final stage would bring rows no cached prediction covers. `predict_new.py`
takes any movements parquet in `ranking.parquet`'s schema and predicts it with the
shipped recipe **from saved models**: features from the same builder, the tree and the
18 MLPs through `submit.py`'s own code path, the 9 FT-Transformers from their `.pt`
checkpoints (bf16 for the d192 members, as trained), then the recorded weights and the
LIRF override.

**Self-test (2026-09-27): from the raw file it reproduces v36 on all 344,841 rows,
max |diff| 0.** TabNet was taken from its cached predictions for that run. Its five
seeds predate model saving and are being retrained with model and encoder files
(`TABNET_PREFIX=rebuilt_`); once they are, `--tabnet models` makes the pipeline fully
self-contained.

**Done (2026-09-27).** The five TabNet seeds were retrained with model and encoder files
saved, and their predictions are **bitwise identical** to the originals (same best epochs,
same val RMSE, max |diff| 0.0), so TabNet training is deterministic and the saved models are
the shipped ones. With every member reloaded from its saved model, `predict_new.py`
reproduces v36 on **344,840 / 344,841 rows**. The one exception is a LIRF row off by 1 s:
a rounding boundary under float differences between in-process and reloaded TabNet
inference. **The pipeline can now predict any new movements file end to end without
cached predictions.**

