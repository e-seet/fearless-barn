# Plan 03: Physical Regimes Segmentation (WTC & Flight Plan Record)

**File:** `exploration/plans/03_plan_physical_regimes_wtc_record.md`  
**Phase:** Stage 3 Domain-Physical Conditioning  
**Objective:** Condition the multi-model blending weights on physical aircraft categories and radar record states to isolate extreme tail outliers.

---

## 1. Domain Motivation: Physical Taxiing Heterogeneity

Airfield surface movement is dictated by physical laws and air traffic control separation rules that differ fundamentally across aircraft classes and operational tracking modes:
1. **Wake Turbulence Category (`Heavy+`)**:
   - Heavy (H) and Super Jumbo (J) aircraft (e.g. Boeing 777/747/787, Airbus A350/A380) require wider taxiway clearances, slower pushbacks, longer engine warmup periods, and 2–3 minute mandatory wake turbulence departure spacing behind preceding takeoffs.
2. **Missing Flight Plan Records (`NoRecord`)**:
   - Flights with `has_flight_record == 0` frequently represent unscheduled positioning, towed aircraft, maintenance moves, or cross-midnight radar tracking rollovers.
   - Analysis revealed that the standard deviation of taxi time for `NoRecord` flights is **3,590 s** (nearly 8× that of scheduled flights).
3. **Regional Aviation (`Regional`)**:
   - Regional turboprops and small jets (CRJs, Embraers) operate from specialized satellite aprons with short taxi paths and faster off-block acceleration.

---

## 2. Hypothesis & Architectural Design

### The Tail Distortion Problem at Rome Fiumicino (`LIRF`)
At LIRF, when blending weights were solved globally per airport, the extreme squared errors from a handful of unrecorded flights forced the optimizer to set the Tree blend weight $\alpha \to 0.02$ (favoring the neural network). This unfairly penalized the 23,000 normal passenger flights, which perform significantly better under the tree model ($\alpha \approx 0.35 - 0.45$).

### The Multi-Regime Solution
We designed **`airport_wtc_rec_multimodel`**, which solves blend weights per $(a, r)$ bucket:
- Airport $a \in \{\text{EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM}\}$
- Physical Regime $r \in \{\text{Heavy+}, \text{NoRecord}, \text{Regional}, \text{Standard}\}$

$$\hat{y}_i = \alpha_{a, r} p_{\text{tree}, i} + (1 - \alpha_{a, r}) p_{\text{mlp}, i}$$

If a cell $(a, r)$ contains fewer than 200 samples in the holdout, it automatically falls back to the parent airport-level weight $\alpha_a$, preserving statistical stability.

---

## 3. Empirical Results Across All 3 Sweep Seeds

`airport_wtc_rec_multimodel` was rigorously tested across seeds 42, 43, and 44 under `forward / with_aobt / keep_all`:

| Metric | Seed 42 | Seed 43 | Seed 44 | 3-Seed Mean |
| :--- | :---: | :---: | :---: | :---: |
| **Validation RMSE** | **247.75 s** | **248.57 s** | **252.12 s** | **249.48 s** |
| *Prior (`airport_wtc_multimodel`)* | *248.24 s* | *248.60 s* | *252.16 s* | *249.67 s* |
| *Prior (`airport_operator_multimodel`)* | *248.33 s* | *248.69 s* | *252.22 s* | *249.75 s* |
| **Test RMSE** | **244.42 s** | **241.90 s** | **240.53 s** | **242.28 s** |
| **LB-Weighted Test RMSE** | **232.17 s** | **230.08 s** | **229.07 s** | **230.44 s** |

*Verification:* Selected **#1 on validation across all three seeds** by `compare.py`, establishing a new benchmark before introducing unsupervised clustering.
