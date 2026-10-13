# Plan 11: Post-Alignment Empirical-Bayes Shrinkage Blending & Sub-219s Leaderboard Record

## 1. Executive Summary & Breakthrough Results

Building directly on Stage 10 ([**Plan 10**](file:///c:/Users/Admin/Desktop/eddie/fearless-barn/exploration/plans/10_plan_dual_regime_piecewise_tail_alignment.md)), a fundamental architectural flaw was uncovered in how multi-family model ensembles were constructed in Stages 4 through 10:

### The "Inversion Order" Error in Earlier Stages:
In Stages 4–10, the pipeline executed:
$$\text{GBDT} + \text{Neural Net} \xrightarrow{\text{Blend on raw unaligned predictions}} \text{Ensemble} \xrightarrow{\text{Dual-Regime Tail Alignment}} \hat{y}$$

Because blending weights between Gradient Boosted Trees (`airport_operator_blend`) and Neural Nets (`mlp_antigravity`) were fitted on **raw unaligned predictions**, unaligned GBDT severely underpredicted the extreme tail outliers at Rome Fiumicino (`LIRF`), predicting $\sim 1,000$ s when true taxi duration was $25,000$ s to $87,500$ s. The MSE loss minimization solver heavily penalized GBDT and assigned ~90% weight to MLP across several hubs, despite GBDT being vastly superior on 99.9% of normal operations.

### The Architectural Fix (Post-Alignment Empirical-Bayes Shrinkage Blending):
$$\text{GBDT}, \text{Neural Net} \xrightarrow{\text{Dual-Regime Tail Alignment}} \text{Aligned GBDT}, \text{Aligned Neural Net} \xrightarrow{\text{Stein-Shrunk Blending } (\lambda=10,000)} \hat{y}$$

By aligning both model families *prior* to solving blending weights, GBDT receives its rightful **72% to 95% weight** across all European airports. Regularizing per-airport blending weights with Empirical-Bayes Stein shrinkage ($\lambda = 10,000$) towards the global baseline ($\alpha_{\text{global}} = 0.720$) eliminates small-sample estimation variance and produces unprecedented gains across all benchmarks:

- **Validation RMSE**: Dropped from 239.58 s to **238.88 s** (-0.70 s improvement).
- **Validation LB-Weighted RMSE**: Dropped from 225.80 s to **225.19 s** (-0.61 s).
- **Test RMSE**: Dropped from 228.77 s to **227.16 s** (-1.61 s all-time record!).
- **Test LB-Weighted RMSE**: Dropped from 220.25 s to **218.97 s** (-1.28 s, **breaking the 219s barrier!**).
- **In-Band Test RMSE ($\le 7200$s)**: Dropped to **218.03 s**.
- **LIRF Test RMSE**: Dropped from 409.41 s to **397.80 s** (breaking below 400s at Rome Fiumicino!).

---

### Official Tournament Benchmark Standings (`compare.py`, 30 Arms)

```
ranked on val RMSE (lower is better; never ranked on test):
  1. airport_debiased_multimodel (Stage 11)  val= 238.88 s   test= 227.16 s   LB= 218.97 s   -> [NEW ALL-TIME RECORD]
  2. airport_debiased_multimodel (Stage 10)  val= 239.58 s   test= 228.77 s   LB= 220.25 s
  3. airport_debiased_multimodel (Stage 9)   val= 243.73 s   test= 233.58 s   LB= 223.84 s
  4. airport_multiseed_multimodel            val= 246.65 s   test= 240.04 s   LB= 228.70 s
  5. airport_unsup_multimodel                val= 247.52 s   test= 244.33 s   LB= 232.13 s
  ...
  30. naive_offblock                         val= 481.97 s   test= 484.76 s   LB= 468.10 s
```

---

## 2. Metric Evolution Across All 11 Stages

| Stage / Champion Arm | Val RMSE | Test RMSE | LB-Weighted Test RMSE | In-Band Test RMSE ($\le 7200$s) | LIRF Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 11: Shrinkage Blending (`airport_debiased_multimodel`)** | **238.88 s** | **227.16 s** | **218.97 s** | **218.03 s** | **397.80 s** |
| Stage 10: Dual-Regime Alignment (`airport_debiased_multimodel`) | 239.58 s | 228.77 s | 220.25 s | 218.16 s | 409.41 s |
| Stage 9: Gated Tail Alignment (`airport_debiased_multimodel`) | 243.73 s | 233.58 s | 223.84 s | 218.04 s | 445.82 s |
| Stage 8: Linear Alignment (`airport_debiased_multimodel`) | 245.73 s | 239.57 s | 228.34 s | 218.13 s | 488.35 s |
| Stage 6: Multi-Seed Ensemble (`airport_multiseed_multimodel`) | 246.65 s | 240.04 s | 228.70 s | 218.16 s | 491.61 s |
| Stage 4: Unsupervised Congestion (`airport_unsup_multimodel`) | 247.52 s | 244.33 s | 232.13 s | 227.72 s | 499.50 s |
| Stage 3: Physical Regimes (`airport_wtc_rec_multimodel`) | 247.75 s | 244.42 s | 232.17 s | 228.00 s | 501.20 s |
| Baseline: Pooled LightGBM (`lightgbm`) | 297.78 s | 315.79 s | 288.37 s | 249.99 s | 620.40 s |

---

## 3. Mathematical Formulation & Empirical-Bayes Shrinkage

### A. Affine Invariance of Dual-Regime Alignment
The dual-regime alignment operator $T(p)$ is an affine transformation of the base prediction $p$:
$$T(p) = c_0 + c_1 \cdot p + c_2 \cdot \text{delta\_sched\_sec}$$
Because $T$ is affine, for any convex combination $\alpha p_{\text{gbdt}} + (1 - \alpha) p_{\text{mlp}}$:
$$T(\alpha p_{\text{gbdt}} + (1 - \alpha) p_{\text{mlp}}) \equiv \alpha T(p_{\text{gbdt}}) + (1 - \alpha) T(p_{\text{mlp}})$$
The mathematical equivalence was empirically verified to machine precision ($\max |\Delta| = 0.00000000$ s).

### B. Empirical-Bayes Stein Shrinkage Formulation
To prevent overfitting on smaller airports (e.g. Zurich `LSZH` $n=25,247$ vs Istanbul `LTFM` $n=50,611$), per-airport raw blending weights $\alpha_a$ are shrunk toward the global optimal baseline $\alpha_{\text{global}} = 0.720$ solved strictly on `forward_inner`:
$$\alpha_{\text{shrunk}, a} = \frac{N_a}{N_a + \lambda} \alpha_a + \frac{\lambda}{N_a + \lambda} \alpha_{\text{global}}$$
Where $\lambda = 10,000$ is the regularization parameter chosen by cross-validation on `forward_inner`.

### C. Solved Weights Across European Hubs

| Airport | Holdout Count $N_a$ | Raw GBDT Weight $\alpha_a$ | Shrunk GBDT Weight $\alpha_{\text{shrunk}, a}$ | Neural Net Weight |
| :--- | :---: | :---: | :---: | :---: |
| **EDDF** (Frankfurt) | 42,802 | 0.990 | **0.939** | 0.061 |
| **EDDM** (Munich) | 31,387 | 0.810 | **0.788** | 0.212 |
| **EGLL** (London Heathrow) | 41,618 | 0.965 | **0.918** | 0.082 |
| **EHAM** (Amsterdam Schiphol) | 45,110 | 1.000 | **0.949** | 0.051 |
| **LEBL** (Barcelona) | 33,348 | 0.915 | **0.870** | 0.130 |
| **LEMD** (Madrid Barajas) | 37,309 | 1.000 | **0.941** | 0.059 |
| **LFPG** (Paris CDG) | 43,793 | 0.895 | **0.862** | 0.138 |
| **LIRF** (Rome Fiumicino) | 30,670 | 0.105 | **0.256** | 0.744 |
| **LSZH** (Zurich) | 25,247 | 0.980 | **0.906** | 0.094 |
| **LTFM** (Istanbul) | 50,611 | 0.995 | **0.950** | 0.050 |
| **Global Baseline** | 382,237 | 0.720 | **0.720** | 0.280 |

Notice that on almost all airports (EDDF, EGLL, EHAM, LEMD, LTFM), GBDT is awarded $\ge 91\%$ weight, dramatically lowering prediction variance, while at Rome Fiumicino (`LIRF`), where neural network representations handle non-linear taxi interactions better, the MLP is retained at $74.4\%$.

---

## 4. Verification & Submission v10

- **Protocol Gate Compliance**:
  - `weights_are_val_fitted: false` (all alphas and regression coefficients fitted exclusively on `forward_inner`).
  - Feature fingerprint: `aeade7fee3099c1a` (100% frozen, 0 features added or modified).
  - Pool equality: `d16701419a0d1c8703890a73c854ce9c` ($n=1,387,105$).
- **Submission Parquet**:
  - Target: [`exploration/submissions/PLACEHOLDER_TEAM_v10.parquet`](file:///c:/Users/Admin/Desktop/eddie/fearless-barn/exploration/submissions/PLACEHOLDER_TEAM_v10.parquet)
  - Rows: $n=215,876$, 0 nulls, strictly bounded within $[60, 86400]$ s.
