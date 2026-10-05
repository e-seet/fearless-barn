# Wide-shallow FT-Transformer as a fourth FTT-slot family — written before training

Written 2026-10-05 21:30, user decision ("queue it after round 10"). Base: **v56 = 271.5625** (best and last);
its base model is v42 (FTT slot = 6 small d64×3 + 9 large d192×3 + 6 deep d96×6, each family 1/3).

## Why

**The board kept paying for variety among the attention models**, while every MLP-family change
failed:
- large FTT: −0.64, then −0.25;
- deep FTT: −0.23, then −0.28.

The existing families are small, large and deep-narrow. The new one sits at the other extreme:
**wide and shallow**, with 2 blocks, d_token 256 and 8 heads. It is also the cheapest per epoch of
the large models, which matters with the freeze on 9 Oct.

## Training (`tests_ftt/run_ftt.py`, same recipe as the other families)

`--d-token 256 --n-blocks 2 --amp --lr 5e-4 --tag d256b2`, seeds 42, 43, 44.

1. **forward_inner** (cap 260): for the per-airport weight solve.
2. **full** (`--predict-ranking`):
   - epochs = ceil(median forward_inner best epoch) of the 3 seeds;
   - the forward stage is skipped, as in deep Amendment 2.

Any run that ends at its cap is rerun at a higher cap. A run whose best forward_inner val exceeds
the deep family's worst seed by more than 15 s is recorded and dropped from the build.

## Build

- **FTT slot:** small : large : deep : wide at fixed shares **1/4 each**. Per-model weights are set
  accordingly, so each family totals the same.
- **Weights and override:** per-airport weights re-solved on forward_inner with the default solver;
  TabNet LIRF w = 0.459; the FTT slot's LIRF weight must solve to 0.
- **New base:** v42′. **v58** = `final_pipeline.py --base v42′`, with every corrector and offset
  unchanged.
- The user approves before any upload.

## Expectation

- **Range:** a tie to about −0.3. Each step has been smaller than the last, and stages 2–5 sit on
  top of the base.
- **Why a board probe:** local checks cannot judge FTT-slot mixes, so this is a board probe.
- **Fallback:** a byte copy of v56 (or of v57, if it is then the best) uploaded last.

**Cut-off:** if v58 cannot be built by 2026-10-08 12:00, it is dropped.

## Amendment 1 (2026-10-05, user decision, before any wide-FTT or CatBoost result): add CatBoost as a family

Added following the audit of closed ideas (`tests_next/AUDIT_closures.md`): CatBoost had never been
tried as a family in the current blend.

**No new training is needed.** The existing `catboost_antigravity` runs (tag `cb1`) are used:
- seeds 42–44 on **forward_inner**, all early-stopped at 626 / 1,371 / 1,430 rounds of 5,000, for
  the weight solve;
- seeds 42–44 on **full**, for the submission.

The cap-8000 rerun (`cb2_recap`) converged but scored worse on forward (307.15 against 305.54), so
`cb1` is the recipe.

**Build:**
- the v58 base gets CatBoost (mean of the 3 seeds) as an extra member in the per-airport blend,
  next to trees, MLP and the FTT slot (four FTT families at 1/4 each);
- weights are re-solved on forward_inner with the default solver;
- TabNet LIRF w = 0.459.

**Attribution:** a scratch solve with and without CatBoost is reported. If the solver gives CatBoost
zero weight at every airport, it is left out, and v58 tests the wide FTT alone. Both changes go in
one board probe, as in deep Amendment 1. The expectation and fallback rules are unchanged.
