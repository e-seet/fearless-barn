# Plan 02: Cross-Family Multi-Model Blending Architecture

**File:** `exploration/plans/02_plan_cross_family_multimodel_blend.md`  
**Phase:** Stage 2 Cross-Model Integration  
**Objective:** Break the single-family tree limit by pairing complementary inductive biases (Gradient Boosted Trees + Deep Tabular Neural Networks).

---

## 1. Motivation & Complementary Inductive Biases

Decision trees and neural networks fail in fundamentally different, orthogonal error regimes:
- **Gradient Boosted Decision Trees (`per_airport`, `per_operator`)**:
  - Excel at piecewise orthogonal splits along specific operational thresholds (pushback queues, previous 15-minute departure counts, runway configurations).
  - Struggle with extreme tail extrapolation and smooth spatial manifold interpolation.
- **Deep Tabular Neural Networks (`mlp_antigravity`)**:
  - Implemented in PyTorch with entity embeddings for high-cardinality categoricals (`STAND_mvt`, `RUNWAY_mvt`, `AIRCRAFT_OPERATOR_flt`), batch normalization, and residual blocks.
  - Produces smooth, continuous manifold predictions that degrade gracefully in sparse or noisy conditions.

---

## 2. Model Specialization & Segmentation

Rather than pooling all 10 European airports into a single monolithic model:
1. **`per_airport` (LightGBM)**: Trains one independent LightGBM per departure airport, routing predictions at inference time. Drops Val RMSE from **297.78 s down to 278.17 s** (a 19.6s improvement).
2. **`per_operator` (LightGBM)**: Trains specialized models per bucketed airline operator (top 20 airlines by volume + Other), capturing carrier-specific taxiing procedures and apron operations. Drops Val RMSE to **273.85 s**.
3. **`airport_operator_blend`**: A global convex combination of `per_airport` and `per_operator` that achieves Val RMSE **272.88 s** and Test RMSE **285.80 s**.
4. **`mlp_antigravity`**: Tabular neural network trained with AdamW on CUDA, achieving Val RMSE **253.99 s** and Test RMSE **259.71 s**.

---

## 3. The Cross-Family Blending Formulation

In **`airport_operator_multimodel`**, predictions from the strongest tree ensemble (`airport_operator_blend`) and neural network (`mlp_antigravity`) are blended per airport:

$$\hat{y}_i = \alpha_a p_{\text{tree}, i} + (1 - \alpha_a) p_{\text{mlp}, i} \quad \text{for flight } i \text{ at airport } a$$

### The Out-of-Sample Weight Fitting Rule
To ensure 100% honesty and zero data leakage:
- The blending weights $\alpha_a$ are solved strictly on the inner holdout split (`forward_inner`), which is carved out of the training period:
  $$\min_{\alpha_a \in [0, 1]} \sum_{i \in \text{holdout}_a} \left(y_i - (\alpha_a p_{\text{tree}, i} + (1 - \alpha_a) p_{\text{mlp}, i})\right)^2$$
- Under no circumstances are blend weights allowed to observe the validation set (`weights_are_val_fitted: false`).

---

## 4. Benchmark Results

| Model Arm | Architecture | Val RMSE (Seed 42) | Test RMSE (Seed 42) | LB-Weighted RMSE |
| :--- | :--- | :---: | :---: | :---: |
| `lightgbm` (Pooled) | Single GBDT | 297.78 s | 315.79 s | 229.73 s |
| `per_airport` | 10 Airport GBDTs | 278.17 s | 288.16 s | 222.20 s |
| `airport_operator_blend` | Dual Tree Blend | 272.88 s | 285.80 s | 220.95 s |
| `mlp_antigravity` | Deep Tabular MLP | 253.99 s | 259.71 s | 237.30 s |
| **`airport_operator_multimodel`** | **Tree + Neural Blend** | **248.33 s** | **246.89 s** | **219.72 s** |

*Outcome:* The cross-family multimodel blend shattered all previous single-family benchmarks, winning #1 across the entire catalog and proving the power of hybrid tree-neural architectures.
