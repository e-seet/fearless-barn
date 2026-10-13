# Hyperparameter-diverse MLP ensemble — pre-registration

Written 2026-09-23, **before any code was run**. Plan 14 §4d.

## Question

Seed averaging has flattened: 12→18 seeds bought −0.54 board. Every member is
the same (256,128,64) MLP. Can MLP members with **different hyperparameters**
make errors different enough that swapping some of them in beats adding more
baseline seeds at the **same member count**?

## Why this is not re-litigating a closed test

`TAXI_OUT_DRIVERS.md` §2j records an earlier MLP sweep (residual connections,
wider/deeper, cosine LR), **closed flat against baseline standalone**.
`tests_arch_diversity/RESULT.md` closed the GRU state model as an ensemble
member: residual correlation 0.98 with the MLP, so it was worth one more MLP
seed and no more.

Neither measured MLP *hyperparameter* variants **as ensemble members**. A
variant that ties standalone is exactly the kind that can pay through
decorrelation (TabNet did). **The prior is low:** same features, same family,
and the GRU's 0.98 is a warning. That is why this is a cheap screen.

## Configs (fixed now, chosen for expected diversity, not accuracy)

All use the production recipe unless stated: AdamW, MSE, clip 5, batch 2048,
lr 1e-3, wd 1e-4, dropout 0.1, (256,128,64), patience 6 on pooled forward val.
Cap is **50** for every config, the baseline included; the cap-45 retrain saw
optima up to 33.

| name | change |
|---|---|
| `base` | production recipe (retrained here with the same script and cap, so the comparison is not confounded by the cap-25 `tests_lirf_seeds` runs) |
| `wide` | hidden (1024,), a single wide layer |
| `deep` | hidden (128,128,128,128,64) |
| `reg` | dropout 0.3, weight decay 1e-3 |
| `bigbatch` | batch 8192, lr 2e-3 |

**Seeds:** 42–47 for every config (30 runs).

## Screen (per variant V)

Surfaces: forward val (Sep–Oct, the early-stopping surface) and forward test
(Nov–Dec). Metric: in-band (y ≤ 7200) RMSE of the equal-weight mean.

- **A** = 6 `base` seeds (42–47).
- **B1** = `base` 42–44 + V 45–47. **B2** = `base` 45–47 + V 42–44.

**PASS for V requires all of:**
1. mean(B1, B2) beats A by **≥ 1.0 s** in-band on **test**;
2. both B1 and B2 beat A on test (the swapped control agrees in sign);
3. mean(B1, B2) beats A on val (same sign).

Also reported, descriptively: residual correlation of the V-ens3 against the
base-ens3, compared with base-ens3 against a different base-ens3 (the
"no diversity" reference), and each config's standalone ens3 RMSE. As the FTT
result showed, a member needs **both** to be decorrelated **and** to be close
enough standalone.

**If any V passes:** before anything is built, a strong-partner check at
production scale. The blend's MLP side must beat an equal-count all-`base`
side inside the 2-member tree+MLP blend, with weights solved on
forward_inner. That check needs forward_inner and `full` runs and is only
specified now, not pre-committed.

**If none pass:** close hyperparameter-diverse ensembling alongside the GRU
result.

## Cost

About 2–3 min per run on the fast GPU loop, ~1.5 GPU hours for 30 runs.
Predictions at the best epoch only (`pv`, `pt`), gitignored `.npz`.
