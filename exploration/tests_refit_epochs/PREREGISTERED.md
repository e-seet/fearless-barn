# How many epochs should the `full` MLP refit train? — pre-registration

Written 2026-09-23, **before any code was run**. Plan 14 §4c; `CONVERGENCE_AUDIT.md` §2.

## The question

The 18 `full` MLP members train for a frozen number of epochs:
- seeds 42–51 run 10 epochs, frozen from forward seed 42's best epoch;
- seeds 52–59 run 14 epochs, frozen from forward seed 44's.

The forward optima range from 10 to 21 (median 12.5), so both counts are
arbitrary. `full` has 1.50× forward's rows, and so 1.50× the updates per epoch
at a fixed batch size. The right direction is undetermined.

One piece of existing evidence already points somewhere. Registered best epochs
**fall** as data grows:

| run | rows | best epoch, median (range) |
|---|---:|---|
| forward_inner, 6 months, cap 45 | 1,005,290 | ~23 (17–33) |
| forward, 8 months | 1,387,105 | 12.5 (10–21) |

This is only suggestive: the two val periods differ as well as the row counts.

## Design: two refit analogs, all read off saved per-epoch curves

Each seed trains three stages. Every stage runs for a **fixed 50 epochs**
(1.5 × the largest observed optimum, 33) and saves predictions at every
epoch. The recipe is production `mlp_antigravity` via the fast GPU loop
(`tests_lirf_seeds/train_seeds.py`): (256,128,64), dropout 0.1, AdamW 1e-3
wd 1e-4, batch 2048, MSE, clip 5. Each stage fits its own categorical maps,
medians and scaler on its own training rows, as production does per split.

| stage | train | scored per epoch |
|---|---|---|
| A | Jan–Jun | Jul–Aug |
| B | Jan–Aug | Sep–Oct, Nov–Dec |
| C | Jan–Oct (forward train ∪ val, keep_all) | Nov–Dec |

**The chosen epoch** E for a stage is what production early stopping would
pick from that stage's curve: patience 6 on pooled val RMSE, cap 45. Training
is deterministic given the seed, so the fixed-50 curve is identical to a
patience-6 run up to its stop point.

- **Surface 1 (primary, closest to the real refit):** pick E on B's Sep–Oct
  curve, apply it to C, and score C on Nov–Dec. Row ratio C/B is measured.
- **Surface 2 (corroborating):** pick E on A's Jul–Aug curve, apply it to B,
  and score B on Sep–Oct. Row ratio B/A = 1.38.

**Seeds:** 42–53, all 12.

## Rules compared (each is a per-seed epoch count applied to the target stage)

| rule | definition |
|---|---|
| `ship` | the production analog: seeds 42–51 use seed 42's E, seeds 52–53 use seed 44's E |
| `median` | every seed uses ceil(median E) |
| `div` | every seed uses round(median E ÷ row ratio), the update-matched rule, predicting **fewer** epochs |
| `mul` | every seed uses round(median E × row ratio) |
| `own` | each seed uses its own E (descriptive) |
| `oracle` | the ensemble's best epoch on the target (**descriptive only, never selectable**) |

**Primary metric:** in-band (y ≤ 7200) RMSE of the **12-seed ensemble mean** on
the target period. Production ships an average, not single seeds. Pooled RMSE
is reported but not barred on, because Nov–Dec pooled is dominated by a handful
of date-artifact rows. Per-seed paired deltas are reported as the noise yardstick.

## Decision rule (fixed now, asymmetric on purpose)

1. **Default:** replace the arbitrary 10/14 with the uniform `median`, **unless**
   `median` is worse than `ship` by more than 0.5 s in-band on surface 1. In that
   case keep production as it is.
2. **Directional change:** adopt `div` (or `mul`) instead **only if** it beats
   both `ship` and `median` by **≥ 1.0 s** in-band on **both** surfaces.
3. **Translating to `full`:** the adopted rule is applied with the production
   inputs: the median of the 12 registered forward best epochs (12.5) and
   full's real row ratio of 1.50. Note the extrapolation: the analogs cover
   only 1.25–1.38× more data.

**Adopting anything means retraining the 18 `full` MLP members under a NEW tag.**
v25, v29, v30 and v31 replay the untagged `mlp_antigravity__full` runs, so
overwriting them would make every current submission unrebuildable. The
retrain and any submission built on it need the user's go-ahead.

**Descriptive only:** stage B's Nov–Dec curve against C's shows whether adding
Sep–Oct to training helps at all. This is a sanity check on the refit itself,
not part of the bar.

## Cost

About 25 min per seed (measured on a 2-epoch smoke run, 59 s for all three stages), ~5 GPU hours for
12 seeds, and ~3.4 GB of float32 per-epoch predictions (float16 would overflow: predictions are clipped at 86,400 s, above float16's 65,504) (gitignored `.npz`).
