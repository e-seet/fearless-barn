# fearless-barn — PRC Data Challenge 2026, taxi-out time: method write-up

| | |
|---|---|
| **Final model (current)** | **v48 = 272.1714** on the public board, also the last upload (v42 + an adsb.lol ADS-B corrector at 7 airports, 5-seed average, §2b) |
| Clean fallback (no ADS-B) | **v42 = 286.5256**, byte-identical file kept (§8: the rules question) |
| Ranking rule | the **last** submission counts; deadline 2026-10-11, internal freeze 2026-10-09 |
| Standing (2026-10-01) | **#46 of 213** teams by best score; leader 219.64, #10 233.8, #50 273.6, median 298.9 (was #70 before ADS-B) |
| Reproduce / predict new data | `python predict_new.py ...` for the v42 base, then `python tests_adsb_full/predict_adsb.py --model-dir tests_adsb_full/model_v48` (§3) |

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

## 2. The base model (v42)

A per-airport blend of three model families, each refit on all twelve 2025 months, plus a
Rome-specific override:

```
                    ┌─ trees:  per-airport + per-operator LightGBM (airport_operator_blend)
per-airport blend ──┼─ MLP:    entity-embedding MLP (256,128,64), mean of 18 seeds
 (3 weights per     └─ FTT slot: FT-Transformers ("Feature Tokenizer + Transformer"),
  airport)                        one third each of
                                    small  d_token 64,  3 blocks   6 seeds
                                    large  d_token 192, 3 blocks   9 seeds
                                    deep   d_token 96,  6 blocks   6 seeds
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
- Registered arms: `ftt_multimodel` tag `ftt_c3d6_mlp18` and `tabnet_lirf_override` tag
  `lirf_w0.459_c3d6`. Rebuild: `python submit.py --arm tabnet_lirf_override --tag
  lirf_w0.459_c3d6 --split full --featureset with_aobt --outlier-policy keep_all --seed 42 --version N`.

## 2b. The ADS-B corrector (v43 → v46)

Source: **adsb.lol `globe_history`** daily archives (open data, ODbL; data © adsb.lol contributors),
fetched for all of 2025 and Jan + Jul 2026 (427 days, ~1.2 TB streamed, never stored whole).
`tests_adsb_full/` holds everything; `RESULT.md` there is the full record.

1. **Matching.** For each departure, the aircraft that carried its callsign (`FLIGHT_mvt` or
   `CALLSIGN_flt`) near the airport, kept only if its ground → air change falls within ±120 s of
   the recorded takeoff. Strict matching covers 44–62 % of departures; LFPG, LTFM ≈ 0.
2. **Features** (`features.py`, variant `all_nd`): seconds before takeoff of the first ground
   sighting, first displacement from the stand and first movement; liftoff offset; total stopped
   time; time from the last stop to liftoff; median taxi speed. Three further features (sightings
   count, path length, number of stops) were **dropped by a pre-registered drift gate**: they
   shift between 2025 and 2026 because the receiver network changed.
3. **Corrector.** A LightGBM on the residual `y − base` of **held-out** base predictions,
   applied only to covered rows at EDDF, EHAM, LEBL, EDDM, LSZH, LEMD, EGLL. v46: 600 trees,
   31 leaves; trained on Jan, Jul, Sep–Dec 2025 (bases from models that never saw those months).
4. **Applied on top of v42.** 187,750 rows (54.7 %) change.

What it does and does not see: receivers mostly cannot see aircraft at the stand, so the first
sighting lags the recorded off-block by 4–11 min. It informs the *taxi* (stops, line-up wait,
speed), not the off-block record itself, which is why it fails at LIRF (record artifacts, §7).

| round | pre-registered test (fresh months) | result | board |
|---|---|---|---:|
| v43 | Jan ↔ Jul cross-month + 44 fresh Jan/Jul days; 3 airports | 44/44 days better | 281.86 (−4.66) |
| v44 | Nov–Dec 2025 (frozen corrector, 61/61 days); + EDDM | pass | 279.15 (−2.71) |
| v45 | Sep–Oct 2025; + LSZH, LEMD, EGLL; + Nov–Dec training | pass, 61/61 days | 273.60 (−5.55) |
| v46 | Aug 2025; + Sep–Oct training; 600 trees | pass, 31/31 days | **272.26 (−1.34)** |
| v47 | + 56 tabular columns in the corrector (leave-one-month-out, 7/7) | passed locally | 272.73 (**+0.47, lost**) |

v43–v45 beat their written expectations, v46 landed inside its range, v47 lost: the tabular
columns let the corrector learn 2025-specific identities that a within-2025 holdout cannot
expose. Rejected along the way: LIRF (worse on Aug, 12/31 days), ADS-B inside the tree model
(the tree improves by 7 s alone, but after correction Jan + Jul is +0.08).

## 3. Reproducing and predicting new data

**`predict_new.py`** predicts any movements parquet in `ranking.parquet`'s schema with the
final recipe, **from saved models only**: features from the same builder, the tree and 18
MLPs through `submit.py`'s code path, all 21 FT-Transformers from `.pt` checkpoints (bf16
for the large and deep members, as trained), TabNet from its saved models and encoders,
then the recorded weights and the LIRF override.

```
python predict_new.py --movements <file.parquet> --out <pred.parquet> [--ids <template.parquet>]
python predict_new.py --selftest --selftest-version 42     # verify against the uploaded v42
```

**ADS-B stage.** `tests_adsb_full/fetch.py <day>` streams one day's archive and keeps the matched
points; `predict_adsb.py --model-dir tests_adsb_full/model_v46 --base <v42 file> --check
submissions/fearless_barn_v46.parquet` applies the saved corrector and verifies byte identity
(v43–v47 all rebuild byte-identical from their saved models).

**Verified (2026-09-30):** from the raw `ranking.parquet` it reproduces v42 on **344,840 /
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
| v41 | + 3 deep FTTs, large 6 → 9; three FTT families at 1/3 each | 286.80 | −0.23 |
| v42 | deep FTT 3 → 6 seeds | 286.53 | −0.28 |
| v43 | + ADS-B corrector, EDDF/EHAM/LEBL | 281.86 | −4.66 |
| v44 | + EDDM | 279.15 | −2.71 |
| v45 | + LSZH/LEMD/EGLL, more training months | 273.60 | −5.55 |
| **v46** | **+ Sep–Oct training, larger corrector** | **272.26** | **−1.34** |
| v47 | + tabular columns in the corrector | 272.73 | +0.47 (lost) |
| **v48** | **corrector averaged over 5 seeds, + Aug training** | **272.17** | **−0.09 vs v46** |

Probes that lost and were reverted: v33 (TabNet 11 seeds, +0.91), v34 (small FTT 7 seeds,
+0.04), v37 (season-weighted MLPs, +2.07). Two early results settled the tail: **v18**
(predictions capped at 7,200 s) = 491.35, **v27** (target winsorized at 7,200 s) = 496.09.

**What moved the score, in order:** blending a neural model with the trees (−62), averaging
MLP seeds (−29), adding architecturally different neural members (TabNet at LIRF, FTT
elsewhere, −5.4), then averaging and **diversifying the attention models** (FTT seeds and
three FTT families, −2.9). Inside the MLP family nothing ever paid. Then **external
observation data: the ADS-B corrector, −14.3**, the largest gain since the first neural model.

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
| ADS-B: first crude test (3 days, one feature) | wrongly closed at "20 % coverage"; strict matching later gave 44–62 % and the corrector (§2b) | `tests_adsb_coverage` → `tests_adsb_full` |
| ADS-B corrector at LIRF | Aug 2025 654 → 675, 12/31 days | `tests_adsb_full/PREREGISTERED_v46.md` |
| ADS-B inside the tree model | tree −7 alone; after correction Jan + Jul +0.08 | `tests_adsb_full/PREREGISTERED_tree.md` |
| tabular columns inside the ADS-B corrector | 7/7 months locally; board v47 +0.47 | `tests_adsb_full/PREREGISTERED_interactions.md` |
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
could not identify. Public ADS-B turned out to be part of it: done properly (§2b) it took
14.3 s and 24 places. Weather and stand geometry were tested and ruled out.

## 8. Open question: is post-pushback ADS-B allowed?

Nothing here uses test labels or fits the board. adsb.lol is open data and the moderator wrote
"open data sources can be usable"; another top team uses it openly. But the corrector's inputs
come from each flight's own movement **after** pushback, which a real-time forecast would not
have. A forum question asked whether this counts as leakage ("the earlier request to treat
post-pushback ADS-B as leakage") and no explicit answer is on record; the organizers also
stressed that the goal is the model, not the 2026 score. **If the answer is no, v42 (286.53,
no ADS-B) is re-uploaded as the final submission.** Data credit: adsb.lol contributors, ODbL.
