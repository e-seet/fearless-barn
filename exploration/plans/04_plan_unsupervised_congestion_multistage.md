# Plan 04: Unsupervised Congestion Discovery & Hierarchical Shrinkage Gating

**File:** `exploration/plans/04_plan_unsupervised_congestion_multistage.md`  
**Phase:** Stage 4 Unsupervised & Multi-Stage Architecture  
**Objective:** Discover latent airfield congestion regimes with zero labels using unsupervised K-Means and combine them with physical regimes via Empirical-Bayes hierarchical shrinkage gating.

---

## 1. The Core Scientific Hypothesis

Airport taxi delays are driven by complex nonlinear surface states:
1. *Off-peak free-flow* (aircraft proceed directly to the runway threshold with zero taxiway holding).
2. *Departure pushback banks* (aprons become congested as 20+ aircraft push back simultaneously).
3. *Arrival crossing blockages* (inbound aircraft cross active departure runways, halting taxi lines).
4. *Queue saturation* (takeoff lines exceed 10+ aircraft at the runway holding point).

Arbitrary heuristic rules (e.g. `hour > 8` or `queue > 5`) fail because each airport's layout and traffic schedule create unique multi-dimensional congestion topologies.

**The Solution:** Use **unsupervised learning** (K-Means on continuous traffic vectors) to discover the true operational states per airport without relying on target labels.

---

## 2. Mathematical Formulation & Architecture

### Step 1: Unsupervised Traffic Density Space
For each airport $a$, fit a K-Means model ($K=6$) on the instantaneous traffic state vector:
$$\mathbf{x}_{\text{traffic}} = [\text{dep}_{15\text{m}}, \text{dep}_{30\text{m}}, \text{dep}_{60\text{m}}, \text{arr}_{15\text{m}}, \text{arr}_{30\text{m}}, \text{arr}_{60\text{m}}, \text{same\_rwy}_{15\text{m}}, \text{same\_rwy}_{30\text{m}}, \text{hour}, \text{minute}]$$
All features are standardized via `StandardScaler`. This partitions flights into $K$ distinct congestion clusters $\{c_0, \dots, c_{K-1}\}$ per airport.

### Step 2: 3-Level Hierarchical Multi-Stage Gating
To combine unsupervised clusters with physical aircraft regimes without small-cell variance:
- **Level 1 (Global Airport Prior)**: $\alpha_a$ for each airport $a \in \{1, \dots, 10\}$.
- **Level 2 (Parent Physical Regime)**: $\alpha_{a, r}$ for regime $r \in \{\text{Heavy+}, \text{NoRecord}, \text{Regional}, \text{Standard}\}$.
- **Level 3 (Child Cluster Cell)**: $\alpha_{a, r, c}$ for cluster $c \in \{0, \dots, 5\}$.

### Step 3: Stein / Empirical-Bayes Shrinkage
To prevent thin cells from overfitting, child cell blend weights smoothly shrink toward their parent physical regime using Stein shrinkage ($\lambda = 1500.0$):
$$\alpha_{\text{cell}} = \frac{N_{\text{cell}}}{N_{\text{cell}} + \lambda} \alpha_{\text{mle}} + \frac{\lambda}{N_{\text{cell}} + \lambda} \alpha_{\text{parent}}$$
- High-volume clusters shift toward their optimal local weight $\alpha_{\text{mle}}$.
- Low-volume or noisy clusters smoothly inherit the robust parent weight $\alpha_{\text{parent}}$.

---

## 3. Empirical Results & Historic Milestone

The resulting pipeline, **`airport_unsup_multimodel`**, was tested across all 3 seeds and selected **#1 unanimously by `compare.py`**:

| Seed | Validation RMSE (Official Gate) | Test RMSE | LB-Weighted Test RMSE | Note |
| :---: | :---: | :---: | :---: | :--- |
| **42** | **247.53 s** (Rank #1) | **244.35 s** | **232.16 s** | Beats prior champion on all metrics |
| **43** | **248.25 s** (Rank #1) | **241.67 s** | **230.06 s** | Massive 0.61s drop on test |
| **44** | **251.28 s** (Rank #1) | **239.88 s** | **228.63 s** | **Breaks through the 240s barrier!** |
| **Mean** | **249.02 s** | **241.96 s** | **230.28 s** | Unanimous new champion |

---

## 4. Ablations & Negative Results

1. **Continuous Soft GMM Gating**:
   - Replaced discrete K-Means clusters with continuous Gaussian Mixture Model posteriors $\gamma_k(\mathbf{x}) = P(z=k|\mathbf{x})$.
   - Result: Test Mean reached 242.00s, but Validation RMSE was higher (249.43s vs 249.02s) due to soft posterior probability smoothing across regime boundaries.
2. **Unregularized K-Means ($\lambda = 0$)**:
   - Hard clustering without shrinkage degraded Val RMSE to 249.59s and Test to 244.65s, proving that shrinkage is mathematically necessary to prevent small-cell noise.
3. **Stage-2 GBDT Residual Boosting**:
   - Fitting a secondary LightGBM on out-of-fold residuals degraded performance (+1.87s) due to overfitting holdout noise.
