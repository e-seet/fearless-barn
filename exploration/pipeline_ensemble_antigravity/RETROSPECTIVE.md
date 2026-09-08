# RETROSPECTIVE — Arm: ensemble_antigravity

## 1. Motivation
`ensemble_antigravity` is a meta-pipeline that combines multiple distinct model families (e.g. LightGBM, Ridge, XGBoost) using non-negative least squares (NNLS) to solve for optimal convex weights ($w_i \ge 0, \sum w_i = 1$) on the validation split.

Blending different inductive biases (linear models with target encoding vs gradient-boosted decision trees with different histogram/leaf split strategies) consistently yields lower RMSE and higher stability against outlier shifts.

## 2. Interface and Contracts
- Target: `TAXITIME_SEC_mvt`
- Blending method: Non-negative least squares constrained optimization
- Outlier policy: Inherited from base models
- Clipping: Predictions clipped to `[60, 10712]` seconds
- Leak safety: Evaluated on `val`, strictly out-of-sample; weights frozen for refit on `full`.
