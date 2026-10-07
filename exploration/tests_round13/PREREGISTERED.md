# Round 13: correctors trained on a base weighted like v59 (with CatBoost) — pre-registration

Written 2026-10-07, before any model is fitted. Board: **v61 = 269.9888** (best and last).

## Why

- **Round 12 / v61:** training the stage-2 corrector on residuals of a base weighted like the shipped
  blend paid −0.48 on the board, more than expected. That base used v42's 3-member weights.
- **The real v59 base is weighted differently:** it has 4 members, with CatBoost carrying large
  weights (LFPG 0.55, EGLL 0.40, EDDF 0.36, LEBL 0.35).
- **The LFPG corrector** (stage 5) was trained on the equal-weight base.

## Realistic-4 base on each held-out surface

- `tests_ftt_wide/final_weights_v59_s4.json` per-airport weights over [FTT, tree, MLP, CatBoost].
- FTT = the held-out small FTT used in rounds 10–12; the wide and large FTTs have no held-out
  predictions on every surface.
- CatBoost = the held-out `cb1` predictions of each surface, averaging the seeds that exist:
  seasonal_full 42; forward_inner 42–47; forward 42–47.
- Fixed in advance; nothing is fitted on the test months.

## Tests (leave one month out over Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025; all applied to the realistic-4 base)

**Stage 2** (7 airports, K2 setting, seeds 42–46):
- **B** = trained on the round-12 base (v42 weights), as in v61;
- **C** = trained on the realistic-4 base.

**Stage 5** (LFPG rows with a track and a record; the v56 setting):
- **L0** = trained on the equal-weight base, as in v56;
- **L1** = trained on the realistic-4 base.

**PASS (C against B; L1 against L0), each on its own:**
- pooled RMSE over all rows better;
- better in at least 5 of 7 months (for L1: at least 2/3 of the months where it changes rows);
- still better with the 5 largest-gain rows removed.

## Build

- **v62** = `final_pipeline.py` on the v59 base, with the passing correctors swapped in
  (stage 2 `model_v62` and/or LFPG `model_v62_lfpg`).
- The expectation is written before upload. If v62 scores above 269.9888, v61 is re-uploaded last.
- **If nothing passes:** v61 stays.
