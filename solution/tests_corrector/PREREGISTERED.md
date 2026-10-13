# Month-fold residual corrector — pre-registration

Written 2026-09-30, before any run. Source: the competition forum ("a corrector trained on
out-of-fold base predictions, folds by month not by day, did more for us than new features").
**Prior: low.** This repo closed a stage-2 residual booster (plan 04: +1.87 s, overfit holdout
noise) and a per-row learned stacker (`TAXI_OUT_DRIVERS.md` §2v), with different setups.

**Base** on each surface: the shipped recipe's local analogue, a 3-member per-airport blend
[FTT (6 small), tree, MLP], weights solved on forward_inner, plus the LIRF TabNet override
(w = 0.459).

**Corrector:** LightGBM (L2, num_leaves 15, min_data_in_leaf 500, learning_rate 0.03,
feature_fraction 0.8, bagging 0.8, lambda_l2 10) predicting the residual y − base from the
row's features plus the base prediction. Trained on **September 2025** forward-val rows,
early-stopped on **October 2025** (patience 100, cap 3,000). Corrected = base + corrector.

**Bar (all must hold):** the corrected prediction lowers **in-band and pooled** RMSE on
**Nov–Dec 2025** (forward test) **and** on **Jan + Jul 2025** (seasonal_full test, whose base
comes from seasonal-trained members), with no single row carrying the Jan + Jul gain.
**PASS →** a production corrector trained on all available out-of-sample months, and a board
probe shown to the user first. **FAIL →** closed with plan 04 and §2v.
