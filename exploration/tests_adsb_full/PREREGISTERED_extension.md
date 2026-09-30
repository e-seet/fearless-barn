# ADS-B, next step: fresh-season corroboration and more airports — pre-registration

Written 2026-09-30, after v43 was built (not yet uploaded), **before any Nov–Dec 2025 ADS-B day is
fetched**.

## Why

- v43 corrects only EDDF, EHAM and LEBL.
- Four more airports have good 2026 coverage: EDDM 96–97 %, LSZH 92–93 %, EGLL 74–97 %, LEMD 58–94 %.
- They were left out because 2025 coverage there was patchy, or the 18-day test was mixed in one
  direction.
- Every Jan + Jul 2025 day is now used, so a fresh surface is needed: **Nov–Dec 2025**. It is the
  forward-split test, and the tree, MLP-12 and FTT models were trained on Jan–Aug only.

## Surface

- **Days:** all 61 days of Nov–Dec 2025, fetched with the fixed `fetch.py`.
- **Base:** `(tree + mlp12 + ftt) / 3`, from `tree_forward_test`, `mlp12_forward_test` and
  `tests_ftt/preds_seed*_lr1e-3.npz["pt"]` (the forward runs).
- This base differs from the Jan + Jul training base, which also mimics how v43 sits on v42.

## A. Corroboration of the v43 corrector (frozen `model/adsb_corrector.txt`, no refit)

**PASS requires all of:**
- pooled RMSE over all Nov–Dec rows improves;
- at least 2/3 of days improve;
- the gain is not carried by the single best row.

A FAIL does not undo v43's own evidence, but it is reported to the user before (or right after) the
v43 upload.

## B. More airports: EDDM, LSZH, EGLL, LEMD, each on its own

For each candidate airport X:
1. **Jan ↔ Jul (all 62 days):** a corrector trained only on X's covered rows in one month and scored
   on the other month improves X's RMSE, in both directions.
2. **Nov–Dec:** a corrector trained on all 62 Jan + Jul days at X, frozen, improves X's RMSE on
   Nov–Dec, and at least 2/3 of the days improve at X.
3. On Nov–Dec it beats the best of 5 permutation controls at X.

**X is added only if all three hold.**

**Then v44:**
- a single corrector is refitted on EDDF, EHAM, LEBL and the added airports (all 62 days);
- its Nov–Dec pooled RMSE must be ≤ the frozen v43 corrector's;
- the same file format and gates as v43 apply, with the expectation written before any upload.

Parameters and features are as in v43 (`all_nd`, the same LightGBM settings). LIRF, LFPG and LTFM
stay excluded.
