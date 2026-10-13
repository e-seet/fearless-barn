# RETROSPECTIVE — Arm: xgboost_antigravity

## 1. Motivation
`xgboost_antigravity` introduces an alternative gradient boosted decision tree implementation using XGBoost (v3.4.0) with histogram-based split finding (`tree_method='hist'`) and native categorical handling (`enable_categorical=True`).

LightGBM and XGBoost construct histogram bins and manage categorical partitions with different heuristics and regularization parameters (`min_child_weight`, `reg_lambda`, `max_depth` vs `num_leaves`). This provides tree-based diversity against `pipeline_lightgbm`.

## 2. Interface and Contracts
- Target: `TAXITIME_SEC_mvt`
- Metric: L2 RMSE (`reg:squarederror` objective, `rmse` evaluation metric)
- Outlier policy: Adheres to `config.OUTLIER_POLICIES` (`winsor_p9999` by default)
- Clipping: Output predictions clipped to `[60, 10712]` seconds
- Leak safety: Reads only permitted features derived from ranking-available columns.
