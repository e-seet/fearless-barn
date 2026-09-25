# Larger FT-Transformer (d_token 192) — pre-registration

Written 2026-09-26, **before any d_token-192 seed was trained** (a 2-epoch timing run
on throwaway seed 99 was deleted). v35 = 287.9247 is the current best **and last**
submission; ranking uses the last submission.

## Why

FTT is the model family that paid most recently on the board (v30 −3.9, v32
−1.79), but standalone it trails the MLP ensemble by ~4–6 s, so the blend gives it
only partial weight. More copies of the small FTT stopped helping at 6 seeds
(Jan + Jul +0.06 at 9; v34 tie). This tests a *better* FTT: d_token 192, the
paper's default, instead of 64. Everything else is unchanged: 3 blocks, 8 heads,
lr 1e-3, wd 1e-5, batch 1024, patience 10, same features and data pipeline.
**bf16 autocast** (`--amp`) is used for the large model only; timing showed 44 s
per epoch against 83 s in fp32, the same speed as the small model.

## Training (`tests_ftt/run_ftt.py --d-token 192 --amp --lr 1e-3`, tag `d192`)

| split | seeds | cap | purpose |
|---|---|---|---|
| forward_inner | 42, 43, 44 | 260 (as small FTT) | weight solve |
| forward | 42, 43, 44 | 175 (as small FTT) | check |
| seasonal_full | 42, 43, 44 | 200 (as small FTT) | **decisive check (Jan + Jul)** |
| full | 42, 43, 44 | fixed = ceil(median forward best epoch of the three d192 seeds) | build only |

A capped run is unresolved and is rerun at a higher cap. **A diverged run** (val
NaN, or best val > 400) is rerun at lr 5e-4; this is the only allowed change.

## Candidates (FTT slot of the 3-member per-airport blend [FTT, tree, MLP])

- **C1:** FTT slot = mean of the 3 d192 seeds.
- **C2:** FTT slot = mean of all 9 FTT models (6 small + 3 d192, equal weight per model).
- **Comparator:** today's FTT slot, the 6 small seeds (v32/v35).

For each, per-airport weights are re-solved on forward_inner (the same MLP partner
as `tests_ftt/check_6seed.py`), frozen, and applied.

## Bar (in-band = y ≤ 7200; all must hold)

1. in-band **lower** than the comparator on forward val, forward test and
   **seasonal_full Jan + Jul**;
2. Jan + Jul pooled not worse than the comparator by more than 2.0 s;
3. FTT's weight at LIRF stays 0 (LIRF identical); otherwise halt.

**If both pass,** C1 is chosen (simpler) unless C2 beats C1 on all three in-band
surfaces. **PASS →** register the arm and TabNet override (w = 0.459, as v35),
check in scratch against v35, show the user before `submit.py`. The build's
board result will be compared with v35; if it is worse, v35 is rebuilt
byte-identically and uploaded last. **FAIL →** closed.

## Amendment 1 (2026-09-26 01:15): the pre-registered divergence rule fired

At lr 1e-3, forward_inner seeds 42 and 43 ended with best val **460.35 (epoch 41)**
and **452.82 (epoch 55)**, against 388.5 / 386.5 for the small FTT on the same
seeds (best epochs 152 / 171). Both exceed the pre-registered best-val-above-400
rule. The early peak followed by decline is the signature of a learning rate too
high for the larger model (the paper uses 1e-4 at this size). As pre-registered,
**every d192 run is redone at lr 5e-4 under tag `d192lr5`.** Nothing else changes.
The queue was stopped before any forward or seasonal_full d192 result existed. The
two lr-1e-3 files (`preds_seed{42,43}_forward_inner_d192.npz`) are kept as
evidence and not used. The bar and candidates are unchanged.
