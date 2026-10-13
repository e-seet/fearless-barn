# Plan 06: Multi-Seed Variance Reduction & Airport-Specialized Neural Networks

## Executive Summary

Following our forensic audit of the 328,009 test flight errors (which revealed severe 1% Pareto concentration where 1.0% of flights drive 45.8% of total SSE, and LIRF alone drives 29.5% of total loss), we hypothesized two orthogonal architectures:
1. **`airport_multiseed_multimodel`**: An out-of-sample variance-reduction ensemble combining the champion `airport_unsup_multimodel` across random seeds (42, 43, 44).
2. **`per_airport_mlp`**: A fleet of 10 dedicated PyTorch Tabular MLPs trained on NVIDIA CUDA with localized stand/runway entity embeddings.

---

## 1. Theoretical Foundation & Architecture

### Strategy 1: Multi-Seed Ensembling (`airport_multiseed_multimodel`)
- **Mathematical Principle**: Gradient boosted trees (via feature and row subsampling) and neural networks (via stochastic initialization and batch ordering) exhibit estimation variance across random seeds. For any unbiased or low-bias estimator with independent estimation variance $\sigma^2$:
  $$\text{Var}\left(\frac{1}{M} \sum_{m=1}^M \hat{y}_m\right) = \frac{\sigma^2}{M}$$
  Because Mean Squared Error decomposes as $\text{MSE} = \text{Bias}^2 + \text{Variance}$, averaging predictions across seeds 42, 43, 44 strictly reduces MSE without altering the expectation.
- **Leakage Immunity**: Zero free parameters are fitted on validation or test labels. The combination is a pure uniform expectation $\hat{y}_{\text{ens}} = \frac{1}{3}(\hat{y}_{42} + \hat{y}_{43} + \hat{y}_{44})$.

### Strategy 2: Per-Airport Specialized Neural Networks (`per_airport_mlp`)
- **Motivation**: Airport physical layouts (stand coordinates, runway networks, apron geometries) are completely disjoint. Just as LightGBM gained +20s by transitioning from pooled to `per_airport`, training 10 dedicated PyTorch models on CUDA allows entity embeddings (e.g., `adep_stand`, `stand_runway`, airline operators) to specialize without interference from foreign airports.
- **Vectorized Architecture**: Built with GELU activations, BatchNorm1d, Dropout(0.1), AdamW optimizer, and Cosine Annealing learning rate schedules. Vectorized tensor slice batching enables direct C++/CUDA memory streaming on RTX 3080 Ti.

---

## 2. Experimental Results & Tournament Standings

Across the official tournament evaluation protocol (`forward / with_aobt / keep_all`):

| Rank | Model Arm | Val RMSE | Test RMSE | LB-Weighted RMSE | In-Band RMSE | Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **#1** | **`airport_multiseed_multimodel`** | **246.65 s** | **240.04 s** | **228.70 s** | **218.16 s** | **New All-Time Tournament Record** |
| #2 | `airport_unsup_multimodel` | 247.52 s | 244.33 s | 232.13 s | 219.37 s | Previous 3-seed champion |
| #3 | `airport_wtc_rec_multimodel` | 247.75 s | 244.42 s | 232.17 s | 219.43 s | 3-way operational gating |
| #4 | `airport_wtc_multimodel` | 248.24 s | 246.06 s | 233.41 s | 219.67 s | Wake turbulence gating |
| #5 | `airport_operator_multimodel` | 248.33 s | 246.89 s | 234.34 s | 219.72 s | Airport-operator blend |
| ... | ... | ... | ... | ... | ... | ... |
| #10 | `mlp_antigravity` (pooled) | 253.99 s | 259.71 s | 245.54 s | 237.30 s | Global neural baseline |
| #19 | `per_airport` (LightGBM) | 278.17 s | 288.16 s | 268.42 s | 222.20 s | Single tree baseline |
| #27 | `per_airport_mlp` | 388.07 s | 412.66 s | 371.89 s | 278.26 s | 10 dedicated PyTorch MLPs |

---

## 3. Deep Forensic Analysis of `per_airport_mlp`

Why did `per_airport_mlp` excel on standard hubs but degrade on extreme outliers?
- **Per-Airport Performance**:
  - Amsterdam Schiphol (`EHAM`): **171.59 s** Val / **199.88 s** Test (beats pooled MLP by ~40s!)
  - Frankfurt (`EDDF`): **190.54 s** Val / **217.71 s** Test (competitive with tree models)
  - Madrid Barajas (`LEMD`): **207.53 s** Val / **207.88 s** Test (excellent consistency)
- **The Outlier Hub Failure (`LIRF`)**:
  - At Rome Fiumicino (`LIRF`): Val RMSE exploded to **1128.92 s** / Test RMSE **1226.23 s**.
  - **Causal Mechanism**: LIRF contains 50 of the 61 extreme outliers ($y > 7200$s) in the entire dataset. When training a dedicated MLP on only $N=108,499$ rows at LIRF, the network lacks the sample density in the deep tail to regularize extrapolations. Under L2 loss (squared error), a single misprediction of an extreme tail flight inflates SSE by millions.
  - In contrast, pooled `mlp_antigravity` leverages the full $N=1,387,105$ dataset to constrain tail representations, and our hierarchical Bayesian shrinkage ($\lambda = 2000.0$) in `airport_unsup_multimodel` pulls noisy cell predictions toward robust parent priors.

---

## 4. Key Takeaways & Production Deployment

1. **Definitive Champion**: `airport_multiseed_multimodel` is crowned the official #1 model across all 29 tournament arms, achieving:
   - **Val RMSE: 246.65 s**
   - **Test RMSE: 240.04 s**
   - **LB-Weighted Test RMSE: 228.70 s** (breaks the 230s barrier)
   - **In-Band Test RMSE: 218.16 s**
2. **Official Submission Generated**: `PLACEHOLDER_TEAM_v6.parquet` generated and validated against all 6 competition gates.
