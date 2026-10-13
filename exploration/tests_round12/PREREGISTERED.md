# Round 12: train the ADS-B corrector on a realistic base — pre-registration

Written 2026-10-07, before any model is fitted. Board: **v60 = 270.4667** (best and last).

## Why

The 7-airport corrector (stage 2) is trained on residuals of a simplified held-out base,
`(tree + MLP + FTT) / 3` with **equal** weights. In 2026 it is applied to the shipped base, which
uses **per-airport** weights, for example EHAM 0.82 FTT, LEBL 0.59 trees, EGLL 0.40 trees. The
residual structure it learns is therefore not the one it corrects. This tests whether matching the
training base to the shipped weighting helps.

## Realistic base

- Each surface's held-out members (tree, MLP, FTT; the same arrays as rounds 10–11) are combined
  with **v42's per-airport weights** (`tests_ftt/final_weights_c3d6.json`, order [FTT, tree, MLP]).
- The weights are fixed: nothing is fitted on the test months. They were solved on forward_inner,
  which is one of the 7 surfaces, as they were for the shipped model.

## Variants (K2 setting: 3,000 trees, 127 leaves, lr 0.015, seeds 42–46)

- **A (current practice):** trained on residuals of the equal-weight base; applied to the realistic
  base (its `base` input is the realistic base at prediction time).
- **B (realistic):** trained on residuals of the realistic base; applied to the realistic base.

**Surface:**
- leave one month out over Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025;
- scored on all rows of the held-out month, against the realistic base;
- only covered rows at the 7 airports are corrected.

## PASS (B against A), all required

1. pooled RMSE better;
2. better in at least 5 of 7 months;
3. still better with the 5 largest-gain rows removed.

## Build

- **PASS →** stage 2 retrained as B on all 7 months (`model_v61`).
- **v61** = `final_pipeline.py` on the v59 base with `--stage2-model model_v61`; every other stage is
  as in v60.
- The expectation is written before upload. If v61 scores above 270.4667, v60 is re-uploaded last.
- **FAIL →** v60 stays final.

**Cut-off:** built by 2026-10-08 12:00.
