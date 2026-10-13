# Round 10: three refinements of the 7-airport ADS-B corrector — pre-registration

Written 2026-10-05, before any model is fitted. Board: **v56 = 271.5625** (best and last).

**Why these three:**
- The last capacity step (v46: 600 trees, 31 leaves) gained −1.09 on Aug locally and helped on the
  board. Capacity has not been pushed further.
- A forum team reported that "a second model family on the same inputs" helped them. Our corrector
  is a single family (LightGBM).
- The corrector sees only the blended base. The disagreement between the base's members (trees, MLP,
  FT-Transformer) is information it does not have.

## Reference A

- The v48 corrector design: 7 airports; the v46 inputs; 600 trees, 31 leaves, learning rate 0.03;
  seeds 42–46 averaged.
- **Leave one month out** over Jan, Jul, Aug, Sep, Oct, Nov and Dec 2025 (the held-out bases of
  rounds 5–9).
- Scored on all rows.

## Variants

- **K (capacity):** as A with 1,500 trees, 63 leaves, learning rate 0.02, min_data_in_leaf 50,
  seeds 42–46.
- **F (second family):** the average of A and a CatBoost regressor on the same inputs and rows
  (depth 8, 2,000 iterations, learning rate 0.05, RMSE loss, airport as a categorical, seeds 42–46
  averaged).
- **M (member disagreement):** as A, plus three inputs: tree − base, MLP − base, FTT − base, from the
  held-out member predictions of each surface.

## Bars (each against A, all required)

1. pooled RMSE better than A;
2. better in at least 5 of 7 months;
3. still better with the 5 largest-gain rows removed;
4. **M only:** the three new inputs pass the drift gate. For each input, the standardized mean
   difference between 2025 (the seasonal_full Jan + Jul test bases) and 2026 (v42's members on the
   ranking set), in Jan and in Jul, must be at most 0.2.

K and F change no inputs, so no permutation control applies. For M, a permutation control (the
three inputs shuffled among the corrected rows) must also lose to M.

## Build

- **Combinations:** if more than one variant passes, the combination of the passing ones is used
  only if it beats each passing variant pooled in the same LOMO run; otherwise the best single one.
- **v57** = v56 with the 7-airport corrector (stage 2) replaced by the winner, trained on all
  7 months.
- **What stays fixed:** the weather offsets, LIRF and LFPG stay exactly as in v56.
- **Built through `final_pipeline.py`,** with the new model directory, so the pipeline stays the
  single source.
- **Before upload:** the expectation is written down. If v57 scores above 271.5625, v56 is
  re-uploaded last.
- **If nothing passes:** v56 stays.

**Cut-off:** build by 2026-10-07.
