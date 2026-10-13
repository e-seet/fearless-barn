# MLP backbone upgrades: PLE numeric embeddings and TabM — pre-registration

Written 2026-09-25, **before any code was run**. v32 = 287.9898 is the safe final;
this screen can only add to it.

## Why these two

- **PLE** (piecewise-linear encoding of numeric features; Gorishniy et al. 2022).
  Every numeric feature currently enters as one standardized scalar through
  BatchNorm. Several are extremely heavy-tailed (`delta_sched_sec` spans −74,644
  to +270,508 s), so standardization squeezes most rows into a narrow band. PLE
  expands each feature into a learned vector over its quantile bins, a different
  axis from the closed MLP variants, which changed capacity or regularization
  (`tests_hp_diversity`, `TAXI_OUT_DRIVERS.md` §2j). The grep found no prior use.
- **TabM** (BatchEnsemble MLP, "TabM-mini" form; Gorishniy et al. 2024). One
  network holds k = 32 members that share every weight except a per-member
  multiplicative adapter on the input, with per-member heads, and averages them.
  The board has twice paid far more for variance reduction than local surfaces
  predicted (MLP 1 → 3 seeds −18, FTT 3 → 6 seeds −1.79), and TabM builds that
  averaging into each model.

## Configs (all others = production `mlp_antigravity` recipe)

(256, 128, 64), dropout 0.1, AdamW lr 1e-3 wd 1e-4, batch 2048, MSE, clip 5,
patience 6 on pooled val, **cap 50**. Fast GPU loop, same data pipeline as
`tests_hp_diversity/run_configs.py`.

| name | change |
|---|---|
| `base` | production MLP (retrained here, same script and cap) |
| `ple` | numeric features → PLE (16 quantile bins per feature, fit on train) → per-feature Linear(16 → 16) + ReLU, flattened; replaces the numeric BatchNorm path. Categorical embeddings unchanged |
| `tabm` | TabM-mini, k = 32: input × per-member adapter (random ±1 init), shared backbone, per-member heads; loss = mean of per-member MSE; prediction = member mean |
| `tabm_ple` | both |

**Seeds:** 42–47 for every config.

## Bar (per variant V; metric = in-band RMSE, y ≤ 7200, of the 6-seed mean)

**Stage 1, forward** (val Sep–Oct is the early-stopping surface; test Nov–Dec):
1. V's 6-seed mean beats `base`'s by **≥ 1.0 s on test**;
2. same sign on val (< 0);
3. ≥ 4 of 6 paired seeds improve standalone on test.

**Stage 2, seasonal_full** (val Dec; test **Jan + Jul**, the board's months). Run
for every config regardless, but only the stage-1 passers are judged:
1. V beats `base` by **≥ 1.0 s on Jan + Jul**;
2. same sign on Dec;
3. **LIRF guard:** V's LIRF in-band on Jan + Jul is not worse than `base`'s by more
   than 2.0 s. The MLP is the entire LIRF base in every shipped blend.

A variant is a candidate only if it passes **both** stages. Three leads this week
passed stage 1 and failed stage 2.

**A capped run** (`hit_cap`) is unresolved: rerun at a higher cap, never judged truncated.

**PASS → production proposal to the user, not automatic.** It needs 18
`forward_inner` and 18 `full` seeds of the variant, a per-airport weight re-solve
(tree/MLP and FTT 3-member), a check of TabNet's w at LIRF against the new base,
and a new tag (v32 must stay rebuildable). It is then built as v34 for the user's approval.

**FAIL →** closed alongside the other MLP-family results.

## Cost

About 2–5 min per run (TabM runs 32 members per row), 48 runs, **~2–3 GPU hours** for both stages.
