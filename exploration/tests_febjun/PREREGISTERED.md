# More training months for the ADS-B corrector (Feb–Jun 2025) — pre-registration

Written 2026-10-02, before any model is trained on the new split. Board: v50 = 271.8748 (best and
last); v48 = 272.1714 is the last pure-ADS-B version.

## Why

- **More training months paid every time:** v45 (+Nov–Dec) and v46 (+Sep–Oct) both improved on the
  board by more than expected.
- **Feb–Jun can't be used yet:** the corrector learns from residuals of *held-out* base predictions.
  Feb–Jun 2025 has ADS-B (fetched for the tree test) but no held-out base, since every existing model
  trained on those months.

## New split `holdout_febjun`

| part | months |
|---|---|
| train | Jan, Jul, Aug, Sep, Oct, Nov 2025 |
| val (early stopping only) | Dec 2025 |
| test (the new held-out base) | **Feb, Mar, Apr, May, Jun 2025** |

**Base on Feb–Jun:** `(tree + mlp6 + ftt3) / 3`.
- tree: `airport_operator_blend` seed 42;
- mlp6: `mlp_antigravity`, seeds 42–47, default recipe;
- ftt3: `tests_ftt/run_ftt.py` small FTT, seeds 42–44, `--lr 1e-3 --epochs 180`.

These are the same recipes as the seasonal_full base, so the bases are built the same way. A
regenerated seasonal_full MLP base must match the cached `mlp6_seasonal_full_test` (fairness control
for the prediction code).

## Test

- **Leave one month out** over the same 7 scored months as before (Jan, Jul, Aug, Sep, Oct, Nov, Dec).
- **A** = the v48 corrector design (v46 inputs and parameters, 7 airports, 5 seeds), trained on the
  other 6 of those months.
- **A5** = the same, with the 5 Feb–Jun months added to training in every fold. Feb–Jun is never
  scored.

**PASS requires:**
1. pooled RMSE: A5 < A;
2. A5 < A in at least 5 of the 7 months;
3. A5 is still better with the 5 rows of largest gain removed.

**Drift gate:** no new inputs are introduced, so none applies. Feb–Jun ADS-B coverage is reported.

## Build

- **PASS →** v55 = v42 + the A5 corrector trained on all 12 months, plus v50's two weather offsets
  carried unchanged (frost +30 s, thunderstorm +200 s, on the same 2026 rows). The expectation is
  written before upload; if it scores above 271.8748, v50 is re-uploaded last.
- **FAIL →** closed; v50 stays.

**Cut-off:** v55 must be built by 2026-10-06.
