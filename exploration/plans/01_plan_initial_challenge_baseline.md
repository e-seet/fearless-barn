# Plan 01: Initial Challenge Baseline & Audit Protocol

**File:** `solution/plans/01_plan_initial_challenge_baseline.md`  
**Phase:** Stage 1 Foundation  
**Objective:** Establish competition baseline floors, label audit, time-ordered data splits, and protocol gates.

---

## 1. Problem Definition & Challenge Setup

* **Competition:** PRC Data Challenge 2026 (Eurocontrol & SESAR).
* **Target:** Predict `TAXITIME_SEC_mvt` (taxi-out time in seconds) for **215,876 departure flights** in January and July 2026 across 10 European major hubs.
* **Evaluation Metric:** Pooled Root Mean Squared Error (RMSE):
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
* **Training Dataset:** Historical flight movements across all 12 months of 2025 ($N = 4,167,797$ total movements; $2,085,047$ departures).
* **Target Mathematical Identity:**
  $$\text{TAXITIME\_SEC\_mvt} = \text{MVT\_TIME\_UTC\_mvt} - \text{BLOCK\_TIME\_UTC\_mvt}$$
  Both timestamps are blanked in the evaluation set, requiring genuine predictive modeling.

---

## 2. Key Audit Findings & Constraints

1. **Flown Trajectory Proxy Dominance (`AOBT_3_flt`)**:
   - `AOBT_3_flt` exists on 98.5% of rows in the evaluation set.
   - Using the raw proxy `MVT_TIME - AOBT_3` alone yields **RMSE 384.9 s** on 2025 (vs 546.4 s for the global mean).
   - *Protocol Decision:* Every arm must be evaluated on both `with_aobt` (primary) and `no_aobt` (ablation).
2. **Evaluation Split Composition Asymmetry**:
   - July 2026 in the scored set covers only 3 airports (EDDF, EGLL, EHAM); January covers all 10.
   - *Protocol Decision:* Evaluations must report both pooled RMSE and **Leaderboard-Weighted RMSE** matching the competition test set distribution.
3. **Temporal Split Structure**:
   - `forward` (Canonical Selection Split):
     - Train: January – August 2025 ($N = 1,387,105$, hash `d16701419a0d1c8703890a73c854ce9c`).
     - Validation: September – October 2025 ($N = 381,895$).
     - Test: November – December 2025 ($N = 328,009$).
   - `forward_inner`: Carved from training months for leak-free, out-of-sample blend weight fitting (`weights_are_val_fitted: false`).
   - `full`: All 12 months used strictly for final submission refits.

---

## 3. Initial Baseline Candidate Arms

| Arm | Model Family | Description | Baseline Val RMSE | Baseline Test RMSE |
| :--- | :--- | :--- | :---: | :---: |
| **`naive_offblock`** | Non-parametric | Proxy read + hierarchical fallback median table (stand $\to$ runway $\to$ airport $\to$ global) | 481.97 s | 484.76 s |
| **`ridge`** | Linear | L2-regularized linear model with target-encoded categoricals | 389.60 s | 393.87 s |
| **`lightgbm`** | GBDT | Pooled LightGBM with native categorical handling and L2 objective | 297.78 s | 315.79 s |

---

## 4. Verification Protocol & Quality Gates

To prevent data leakage, overfitting, and synthetic score inflation, three strict verification gates were codified in `compare.py`:
1. **Training Pool Hash Gate**: Every competing arm must train on the identical row pool hash `d16701419a0d1c8703890a73c854ce9c`.
2. **Feature Agreement Gate**: Feature extraction code must match SHA fingerprint `aeade7fee3099c1a`.
3. **Out-of-Sample Weight Fitting**: Any model blending weights must be solved on `forward_inner`, never on the validation set (`weights_are_val_fitted: false`).
