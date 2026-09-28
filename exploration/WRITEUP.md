# fearless-barn — PRC Data Challenge 2026, taxi-out time: method write-up

| | |
|---|---|
| **Final model (current)** | **v41 = 286.8032** on the public board, also the last upload |
| Ranking rule | the **last** submission counts; deadline 2026-10-11, internal freeze 2026-10-09 |
| Standing (2026-09-29) | #70 of 164 teams by best score; leader 221.60, #10 237.0, median 292.4 |
| Reproduce / predict new data | `python predict_new.py --movements <file> --out <pred>` (§3) |

`README.md` and `SUBMISSION.md` are mid-build snapshots and point here. Detailed records:
`plans/` (numbered decision history), `tests_*/RESULT.md` (one per experiment),
`submissions/board_scores.csv` (every board score), `PIPELINE_REVIEW.md`, `TAXI_OUT_DRIVERS.md`.

---

## 1. Task and metric

Predict taxi-out time (seconds from off-block to takeoff) for every departure in the scored
set: **January and July 2026** at ten European airports (EDDF, EDDM, EGLL, EHAM, LEBL, LEMD,
LFPG, LIRF, LSZH, LTFM). Training data: all of 2025, 2,084,659 labelled departures. The
metric is **pooled RMSE** over 344,841 scored rows.

The metric shapes everything. About 0.02 % of rows (taxi-outs above 7,200 s) carry 15–44 %
of the squared error, depending on the period. Under L2 the optimal prediction is the
conditional *mean*, so anything that shrinks the tail (winsorizing, robust losses, prediction
caps) loses (§6).

## 2. The final model (v41)

A per-airport blend of three model families, each refit on all twelve 2025 months, plus a
Rome-specific override:

```
                    ┌─ trees:  per-airport + per-operator LightGBM (airport_operator_blend)
per-airport blend ──┼─ MLP:    entity-embedding MLP (256,128,64), mean of 18 seeds
 (3 weights per     └─ FTT slot: FT-Transformers ("Feature Tokenizer + Transformer"),
  airport)                        one third each of
                                    small  d_token 64,  3 blocks   6 seeds
                                    large  d_token 192, 3 blocks   9 seeds
                                    deep   d_token 96,  6 blocks   3 seeds
        │
        └─ at LIRF only:  0.541 × blend + 0.459 × TabNet (mean of 5 seeds)
```

- **Per-airport weights** are solved on `forward_inner` (train Jan–Jun 2025, holdout
  Jul–Aug), a surface no scored model is fitted on, then frozen. The FTT slot's weight at
  LIRF solves to exactly 0 in every solve, so LIRF is MLP + TabNet only.
- **TabNet's LIRF weight 0.459** is the exact minimum of board error. LIRF's prediction is
  linear in that weight, so board MSE is an exact parabola through three existing uploads
  (v25, v29, v28); v35 landed 0.0001 s from the prediction. It is the one board-fitted
  constant in the model.
- **Training target:** `keep_all`. Every valid row is kept, tail included. Predictions are
  clipped to [60, 86,400] s.
- **Refit:** hyperparameters and epoch counts are frozen from time-ordered splits; `full` has
  no holdout, so nothing is selected on it.
- Registered arms: `ftt_multimodel` tag `ftt_c3_mlp18` and `tabnet_lirf_override` tag
  `lirf_w0.459_c3`. Rebuild: `python submit.py --arm tabnet_lirf_override --tag
  lirf_w0.459_c3 --split full --featureset with_aobt --outlier-policy keep_all --seed 42 --version N`.

## 3. Reproducing and predicting new data

**`predict_new.py`** predicts any movements parquet in `ranking.parquet`'s schema with the
final recipe, **from saved models only**: features from the same builder, the tree and 18
MLPs through `submit.py`'s code path, all 18 FT-Transformers from `.pt` checkpoints (bf16
for the large and deep members, as trained), TabNet from its saved models and encoders,
then the recorded weights and the LIRF override.

```
python predict_new.py --movements <file.parquet> --out <pred.parquet> [--ids <template.parquet>]
python predict_new.py --selftest --selftest-version 41     # verify against the uploaded v41
```

**Verified (2026-09-29):** from the raw `ranking.parquet` it reproduces v41 on **344,840 /
344,841 rows**. The one exception is a LIRF row off by 1 s, a rounding boundary between
in-process and reloaded TabNet inference.

- The five TabNet seeds were retrained with model saving; their predictions are **bitwise
  identical** to the originals, so TabNet training here is deterministic.
- `tests_repro/rebuild_check.py` rebuilds uploads through `submit.py` and records every file
  opened. The untracked files it needs (trained runs, feature cache, FTT/TabNet
  checkpoints) are listed with sha256 in `tests_repro/dependency_manifest.csv`. **They are
  the backup set:** git holds the code and evidence, not the models.

## 4. Validation protocol

| split | train | holdout | used for |
|---|---|---|---|
| `forward` | Jan–Aug | val Sep–Oct, test Nov–Dec | early stopping, model comparison |
| `forward_inner` | Jan–Jun | Jul–Aug | solving blend weights (out of sample) |
| `seasonal_full` | Feb–Jun, Aug–Nov | val Dec, test **Jan + Jul** | the board's months; confirmation before any build |
| `full` | all 12 months | none | the submission refit only; selection forbidden |

Rules that paid for themselves:

- **Pre-register the bar before the number exists** (`tests_*/PREREGISTERED*.md`, `PROBE*.md`).
- **Multi-seed, paired by seed.** MLP single-seed val SD is ~3.5 s.
- **A forward (Nov–Dec) pass needs a Jan + Jul replication.** Six leads passed forward and
  failed on Jan + Jul (ensemble epochs, `deep` MLP, tree seeds, 9 FTT seeds, the large FTT
  alone, season weighting); the one board-tested of them (season weighting, v37) lost 2.07.
- **A capped run is unresolved, not failed.** Three verdicts in this build were first read
  off truncated runs; FTT was nearly closed at cap 60 and later became v30.
- **The local surfaces under-read seed averaging and model variety on the board.** MLP 1 → 3
  seeds: −0.9 local, −18 board. FTT 3 → 6 seeds: −0.13 to −0.30 local, −1.79 board. The
  large FTT mixed in: a local tie, −0.64 board. For these questions a board probe with a
  fallback is the only instrument, and it was used as such, with the expectation written
  down before each upload.
- **Local → board mapping** across models: board ≈ −246 + 2.24 × local val, residual SD ~9.
  Composition checks on disjoint rows are exact: v31 landed 0.0014 s from its prediction.

## 5. Board history: what moved the score

| version | change | board | Δ |
|---|---|---:|---:|
| v2 | pooled LightGBM | 405.75 | |
| v3 | per-airport stack (trees only) | 386.22 | −19.5 |
| v13 | + 1-seed MLP in a per-airport blend | 324.28 | −61.9 |
| v14 → v25 | MLP seeds 3 → 18 (and clean weight solves) | 306.32 → 295.21 | −29.1 |
| v29 | + TabNet override at LIRF (w = 0.55) | 293.67 | −1.5 |
| v30 | + FT-Transformer (small, 3 seeds) as a third member | 291.34 | −3.9 vs v25 |
| v31 | v30 + v29's TabNet override (predicted to 0.0014 s) | 289.78 | −5.4 vs v25 |
| v32 | small FTT 3 → 6 seeds | 287.99 | −1.79 |
| v35 | TabNet LIRF weight → board-optimal 0.459 | 287.92 | −0.065 |
| v36 | FTT slot + 3 large FTTs | 287.29 | −0.64 |
| v39 | large FTT 3 → 6 seeds (share 1/3) | 287.04 | −0.25 |
| **v41** | **+ 3 deep FTTs, large 6 → 9; three FTT families at 1/3 each** | **286.80** | **−0.23** |

Probes that lost and were reverted: v33 (TabNet 11 seeds, +0.91), v34 (small FTT 7 seeds,
+0.04), v37 (season-weighted MLPs, +2.07). Two early results settled the tail: **v18**
(predictions capped at 7,200 s) = 491.35, **v27** (target winsorized at 7,200 s) = 496.09.

**What moved the score, in order:** blending a neural model with the trees (−62), averaging
MLP seeds (−29), adding architecturally different neural members (TabNet at LIRF, FTT
elsewhere, −5.4), then averaging and **diversifying the attention models** (FTT seeds and
three FTT families, −2.9). Inside the MLP family nothing ever paid.

## 6. What did not work (closed, with the evidence)

| idea | result | where |
|---|---|---|
| winsorize / drop the tail in training | `keep_all` wins; board v27 = 496 | `config.py`, `artifacts/clip/` |
| cap predictions | board v18 = 491 | plan 07 |
| prediction floor 60 → 240 s | worse on every surface | `tests_pred_floor` |
| log1p target, Huber, Poisson, quantile ladder | all lose; they de-weight the tail | `tests_log_target`, `TAXI_OUT_DRIVERS.md` §2i, `tests_poisson_objective`, `tests_quantile_ladder` |
| hurdle model (P(tail) × magnitude) | lost twice; 0.02 % base rate | plan 07 |
| LIRF "identity" override (no flight record → scheduled delay) | 100 % right in 2025, **+0.86 on the 2026 board twice** | `tests_lirf_identity`, `tests_tail_study` |
| neighbour / sequence models (GRU, queue attention, retrieval) | residual corr 0.98 with the MLP | `tests_seqrnn`, `tests_arch_diversity`, `tests_retrieval_screen` |
| hyperparameter tuning | 6.5 s for a full sweep vs 27 s for one reasoned constant | `PIPELINE_REVIEW.md` |
| MLP variants (wide, deep, regularized, big-batch) | fail at equal member count | `tests_hp_diversity` |
| MLP backbones: PLE numeric embeddings, TabM, TabM + PLE | all fail on forward | `tests_backbone` |
| season-matched per-month MLP weighting | Jan + Jul +1.70, LIRF +10.5; board v37 +2.07 | `tests_season_weight` |
| transductive pretraining for FTT | +19.8 s worse on the same seed | `tests_ftt_pretrain` |
| refit epoch count / longer training | flat on Jan + Jul, +19 s pooled | `tests_refit_epochs` |
| tree-side seed averaging (1 → 6) | Jan + Jul +0.16 | `tests_tree_seeds` |
| FTT at LIRF, downward-only guard | weight solves to 0 | `tests_ftt_lirf_guard` |
| extra TabNet seeds (5 → 11) | board v33 +0.91 | `tests_tabnet/PROBE_v33.md` |
| ADS-B (adsb.lol) ground sightings as a feature | 20 % coverage (0–3 % at the hubs); LIRF +66 s worse where covered | `tests_adsb_coverage`, `tests_tail_study` |
| METAR weather; stand-to-runway distance (OSM) | inside the noise | `tests_weather_metar`, `TAXI_OUT_DRIVERS.md` §2h |
| exact (tight-tolerance) blend-weight solve | not better out of sample; the default's early stop acts as shrinkage | solver docstring |

A convergence audit (`CONVERGENCE_AUDIT.md` §9) re-checked every closed outlier verdict for
truncated runs; all stand.

## 7. Where the remaining error is, and the leaderboard

On the Jan + Jul 2025 analogue (`tests_tail_diagnosis`), 56 % of squared error is in-band
(≤ 7,200 s) and 44 % in ~150 tail rows, almost all LIRF departures without a flight record
(`tests_tail_study`). In 2025 those rows are structured (the true value equals the scheduled
delay, or ≈ one day), but that pattern **did not transfer to the 2026 board**.

The leader has ~40 % less squared error than v41. **We could not establish where that gap
comes from.** It is not the tail alone: if our board in-band error matches the analogue's
(~250), a perfect tail would reach ~250, not 221. Rank correlates with submission volume
(Spearman −0.64), which points to board probing, something we ruled out. But several top-10
teams reached 224–237 with 22–41 uploads, so there is also a genuine method difference we
could not identify. Public ADS-B, weather and stand geometry were each tested and ruled out
as its source for us.
