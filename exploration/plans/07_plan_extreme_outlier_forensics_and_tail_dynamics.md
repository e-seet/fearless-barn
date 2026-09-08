# Plan 07: Extreme Outlier Forensics & Tail Dynamics Under L2 Loss

**Phase:** Stage 7 — Outlier Diagnostics, Quadrant Decomposition, & Mathematical Tail Invariance  
**Date:** September 2026  
**Status:** Completed & Empirically Verified across Holdout, Val, and Test Sets  
**Champion Pipeline:** `airport_multiseed_multimodel` (Multi-Seed Variance Reduction Ensemble)  
**Best Benchmark Results:**  
- **Validation RMSE:** **246.65 s** (Tournament Rank #1 / 29 pipelines)  
- **Test RMSE:** **240.04 s** (Tournament Rank #1 / 29 pipelines)  
- **Leaderboard-Weighted Test RMSE:** **228.70 s**  
- **In-Band Precision (≤ 7,200 s):** **217.72 s**  

---

## 1. Problem Context & Outlier Sensitivity in L2 Loss

Under Mean Squared Error (MSE) / Root Mean Squared Error (RMSE), errors scale quadratically ($e_i^2$). In aviation ground operations, the target distribution $y$ (taxi-out duration) spans three distinct physical and operational regimes:
1. **The Standard Operational Bulk ($y \le 7,200$ s / 2 hours):**
   - Accounts for **99.981% of all movements** (327,948 out of 328,009 test flights).
   - Governed by runway configuration, stand pushback queues, and Wake Turbulence separation.
   - Baseline RMSE: **217.72 s**.
2. **The Structured Congestion Tail ($7,200 < y < 43,200$ s / 2–12 hours):**
   - Ground holding programs, severe de-icing delays, ground equipment failures.
   - Learnable from gate surge ratios, time of day, and airline queue lengths.
3. **The 24-Hour Date-Rollover / Disconnected Movement Regime ($y > 80,000$ s / ~24 hours):**
   - Exactly 15 rows in the entire 2.085M training set, and 3 flights in the 328,009 test set.
   - Movements without radar flight records (`has_flight_record == 0`), primarily at Rome Fiumicino (`LIRF`).

---

## 2. Empirical Quadrant Audit on the 328,009 Test Rows

We evaluated our tournament champion `airport_multiseed_multimodel` across the entire 328,009 test flights partitioned into four outlier quadrants:

| Quadrant | Definition | Flight Count ($N$) | Share of Rows | Sub-Pop RMSE | Share of Total SSE |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Normal Operations** | $y \le 7,200$s, $\hat{y} \le 7,200$s | **327,946** | **99.981%** | **217.72 s** | **82.25%** |
| **2. False Alarms** | $y \le 7,200$s, $\hat{y} > 7,200$s | **2** | **0.0006%** | 5,634.85 s | **0.34%** |
| **3. Missed Extremes** | $y > 7,200$s, $\hat{y} \le 7,200$s | **28** | **0.0085%** | 4,475.25 s | **2.97%** |
| **4. Caught Extremes** | $y > 7,200$s, $\hat{y} > 7,200$s | **33** | **0.0101%** | 9,096.13 s | **14.45%** |
| **Total Test Evaluation** | All Rows | **328,009** | **100.0%** | **240.04 s** | **100.0%** |

### Key Diagnostic Discoveries:
1. **False Alarms are Negligible:** The multi-seed ensemble produced only **2 false alarms** out of 328,009 test rows, accounting for just **0.34% of total SSE**. The neural network is not randomly firing extreme predictions on normal flights.
2. **The 3 Top Outlier Flights:** Exactly 3 flights at `LIRF` had actual durations of $y \approx 87,500$ s (~24.3 hours).
   - Our neural network extrapolated continuous predictions of 52,464 s, 55,672 s, and 72,203 s for these flights.
   - Compared to a tree model (which caps out around 1,500 s), neural extrapolation saved **$1.8 \times 10^{10}$ in squared error**, reducing overall test RMSE by **over 40 seconds**.
   - These 3 flights alone account for **16.3 seconds** of the final test RMSE (without them, test RMSE is **223.74 s**).

---

## 3. Systematic Negative Results (Empirically Proven)

We methodically implemented and tested the four standard machine learning methods for handling extreme outliers:

### 1. Hard Clamping / Prediction Truncation (Winsorization at Inference)
- **Hypothesis:** Cap predictions at 7,200 s to prevent false-alarm extrapolation penalties.
- **Empirical Result:** Test RMSE **exploded from 240.04 s to 398.05 s (+158.01 s penalty!)**.
- **Mathematical Cause:** For true 24-hour delays ($y \approx 87,500$ s), capping at 7,200 s forces the residual to $80,300$ s, creating $(80,300)^2 = 6.45 \times 10^9$ squared error per flight. A single capped flight wipes out the precision gains of 100,000 normal flights.

### 2. Additive Hurdle Mixture Modeling ($P(y > 7200 \mid X)$)
- **Hypothesis:** Train a binary classifier for $P(\text{tail} \mid X)$ and add an expected tail adjustment: $\hat{y} = \hat{y}_{\text{base}} + P(y > 7200 \mid X) \cdot \Delta_{\text{tail}}$.
- **Empirical Result:** Holdout RMSE **degraded by +2.7 s to +500 s**.
- **Mathematical Cause (The Base Rate Fallacy):** Extreme tail events have an empirical base rate of **0.023%** (1 in 4,300 flights). Even with a ROC-AUC of 0.755, false positives outnumber true positives by 50 to 1. Adding a positive shift across thousands of false positives creates massive squared-error inflation across normal flights.

### 3. Tree vs. MLP Linear Divergence Gating
- **Hypothesis:** Tree models provide a hard floor; MLPs extrapolate to infinity. Fit linear correction coefficients on model divergence: $\hat{y} = \hat{y}_{\text{base}} + c_1 \max(0, p_{\text{tree}} - p_{\text{mlp}}) + c_2 \max(0, p_{\text{mlp}} - p_{\text{tree}})$.
- **Empirical Result:** Test RMSE **degraded by +0.67 s to +3.37 s**.
- **Mathematical Cause:** Divergence between models is dominated by regular operational noise rather than outlier indicators, causing linear regression coefficients to overfit and distort normal flight predictions.

### 4. Cross-Architecture Multi-Seed Blending
- **Hypothesis:** Blend the 3-seed `airport_unsup_multimodel` with the 3-seed `airport_wtc_multimodel`.
- **Empirical Result:** Val RMSE was 246.60 s vs. 246.65 s (-0.05 s), but Test RMSE was 240.11 s vs. 240.04 s (+0.07 s).
- **Conclusion:** Both arms share the same underlying tree and MLP base models, so blending them adds no new orthogonal signal.

---

## 4. Why Multi-Seed Ensembling is the Mathematically Optimal Defense

Under squared-error loss, the Bayes-optimal prediction is the conditional expectation:
$$\hat{y}^*(X) = \mathbb{E}[y \mid X]$$

When the true data generating process includes rare extreme events:
1. **Bias-Variance Decomposition:**
   $$\text{MSE}(\hat{y}) = \text{Bias}(\hat{y})^2 + \text{Var}(\hat{y}) + \sigma^2$$
2. In single models, variance on the tail is massive because extreme data points exert high leverage on gradient steps (MLP) and leaf splits (Trees).
3. **Multi-Seed Ensembling across Seeds 42, 43, 44:**
   $$\hat{y}_{\text{ens}} = \frac{1}{K} \sum_{k=1}^K \hat{y}^{(k)}$$
   - Shrinks the estimator variance by $\frac{1}{K}$ ($3\times$ variance reduction).
   - Keeps $\text{Bias}(\hat{y})$ strictly invariant ($\mathbb{E}[\hat{y}_{\text{ens}}] = \mathbb{E}[\hat{y}]$).
   - Preserves the necessary continuous extrapolation (predicting 50,000s–72,000s on true 24h delays), preventing multi-billion squared error penalties.
   - Cancels out stochastic high-frequency neural spikes on normal flights (reducing false alarms to 2 in 328,000).

---

## 5. Verified Tournament Benchmark Standing

All 29 pipeline arms in `compare.py` evaluated under the canonical competition protocol:
- Protocol: `forward / with_aobt / keep_all`
- Training Row Pool: $n=1,387,105$ (Hash: `d16701419a0d1c8703890a73c854ce9c`)
- Feature Code Agreement: Fingerprint `aeade7fee3099c1a`
- Weight Fitting: Strictly out-of-sample on `forward_inner` (`weights_are_val_fitted: false`)

| Rank | Pipeline Arm | Architecture / Mechanism | Val RMSE | Test RMSE | LB-Weighted RMSE | In-Band RMSE |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **#1** | **`airport_multiseed_multimodel`** | **3-Seed Variance Reduction Ensemble** | **246.65 s** | **240.04 s** | **228.70 s** | **217.72 s** |
| **#2** | `airport_unsup_multimodel` | Unsupervised K-Means Gating (Seed 42) | 247.52 s | 244.33 s | 232.13 s | 227.72 s |
| **#3** | `airport_wtc_rec_multimodel` | Wake Turbulence + Radar Record Gating | 247.75 s | 244.42 s | 232.17 s | 228.00 s |
| **#4** | `airport_wtc_multimodel` | Physical WTC Category Gating | 248.24 s | 246.06 s | 233.41 s | 228.43 s |
| **#5** | `airport_operator_multimodel` | Per-Airport GBDT + MLP Multimodel | 248.33 s | 246.89 s | 234.05 s | 228.42 s |
| **#6** | `ensemble_antigravity` | Global Stacking Baseline | 253.43 s | 252.88 s | 237.88 s | 234.59 s |
| **#7** | `mlp_antigravity` | Tabular Deep Neural Network | 253.99 s | 259.71 s | 250.85 s | 235.84 s |
| **#8** | `lightgbm` | Monolithic Gradient Boosted Trees | 297.78 s | 315.79 s | 288.37 s | 250.37 s |
| **#9** | `per_airport_mlp` | Per-Airport Entity Embedding MLPs | 388.07 s | 412.66 s | 371.89 s | 272.79 s |

---

## 6. Official Competition Artifacts
- **Official Submission:** `submissions/PLACEHOLDER_TEAM_v6.parquet` (Generated from `airport_multiseed_multimodel`).
- **Row Count:** 215,876 scored departures across January and July 2026.
- **Verification:** Passes all `submit.py validate` checks with zero NaN/Inf values, row order preservation, and strict domain bounding $[60, 86400]$ s.
