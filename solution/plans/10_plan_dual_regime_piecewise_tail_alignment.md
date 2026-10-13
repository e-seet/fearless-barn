# Plan 10: Dual-Regime Piecewise Tail Alignment & Sub-229s Record

## 1. Executive Summary & Breakthrough Results

Building directly on Stage 9 ([**Plan 09**](file:///c:/Users/Admin/Desktop/eddie/fearless-barn/solution/plans/09_plan_gated_tail_alignment_and_sub_235s_record.md)), deep forensic audits of residual errors across all 10 competition airports revealed a fundamental physical duality in Eurocontrol telemetry delays at Rome Fiumicino (`LIRF`):
1. **Extreme Overnight Delays ($ds > 50,000$ s / ~14 to 24 hours)**:
   Flights delayed across midnight into the next calendar day. In this subpopulation, Eurocontrol records $y \approx 87,500$ s (24 hours + actual taxi time).
2. **Intermediate Daytime Delays ($25,200\text{s} < ds \le 50,000$ s / ~7 to 14 hours)**:
   Flights delayed substantially during the operational day. Here, the relationship is the exact identity $y \equiv \text{delta\_sched\_sec}$ down to seconds ($\beta_0 = -1.08, \beta_2 = 1.0000$).
3. **Normal Operations ($ds \le 25,200$ s)**:
   Normal flights left completely unperturbed to the multi-seed GBDT + MLP unsupervised ensemble.

In Stage 9, using a single linear threshold at 10,800s conflated these two distinct physical regimes, pulling daytime predictions upwards while underpredicting 24-hour overnight delays.

By decoupling the tail into **Dual-Regime Piecewise Alignment**, both validation and test errors break through historic milestones:

### Official Tournament Benchmark Standings (`compare.py`, 30 Arms)

```
ranked on val RMSE (lower is better; never ranked on test):
  1. airport_debiased_multimodel (Stage 10)  val= 239.58 s   test= 228.77 s   LB= 220.25 s   -> [NEW ALL-TIME RECORD]
  2. airport_debiased_multimodel (Stage 9)   val= 243.73 s   test= 233.58 s   LB= 223.84 s
  3. airport_multiseed_multimodel            val= 246.65 s   test= 240.04 s   LB= 228.70 s
  4. airport_unsup_multimodel                val= 247.52 s   test= 244.33 s   LB= 232.13 s
  ...
  30. naive_offblock                         val= 481.97 s   test= 484.76 s   LB= 468.10 s
```

---

## 2. Metric Progress Across Stages

| Stage / Champion Arm | Val RMSE | Test RMSE | LB-Weighted Test RMSE | In-Band Test RMSE ($\le 7200$s) | LIRF Test RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 10: Dual-Regime Alignment (`airport_debiased_multimodel`)** | **239.58 s** | **228.77 s** | **220.25 s** | **218.16 s** | **409.41 s** |
| Stage 9: Gated Tail Alignment (`airport_debiased_multimodel`) | 243.73 s | 233.58 s | 223.84 s | 218.04 s | 445.82 s |
| Stage 8: Linear Alignment (`airport_debiased_multimodel`) | 245.73 s | 239.57 s | 228.34 s | 218.13 s | 488.35 s |
| Stage 6: Multi-Seed Ensemble (`airport_multiseed_multimodel`) | 246.65 s | 240.04 s | 228.70 s | 218.16 s | 491.61 s |
| Stage 4: Unsupervised Congestion (`airport_unsup_multimodel`) | 247.52 s | 244.33 s | 232.13 s | 227.72 s | 499.50 s |
| Stage 3: Physical Regimes (`airport_wtc_rec_multimodel`) | 247.75 s | 244.42 s | 232.17 s | 228.00 s | 501.20 s |
| Baseline: Pooled LightGBM (`lightgbm`) | 297.78 s | 315.79 s | 288.37 s | 249.99 s | 620.40 s |

---

## 3. Physical & Telemetry Mechanism

### A. Cross-Airport Telemetry Audit
A forensic sweep across all 10 European airports revealed that this telemetry identity is strictly unique to Rome Fiumicino (`LIRF`):
- At `EGLL`, `EHAM`, `LFPG`, `EDDF`, etc., delayed flights without radar records (`has_flight_record == 0` and $ds > 3600$s) still have normal taxi times ($y \approx 1000$s). Their actual taxi duration is uncorrelated with schedule delay ($r \approx 0.05$).
- At `LIRF`, Italian ANSP telemetry defaults $\text{BLOCK\_TIME}$ to $\text{SCHED\_TIME}$ whenever radar track data is missing. Thus:
  $$\text{actual\_taxitime} = \text{TAKEOFF} - \text{BLOCK\_TIME} = \text{TAKEOFF} - \text{SCHED\_TIME} \equiv \text{delta\_sched\_sec}$$

### B. The Dual Regimes at LIRF
1. **Intermediate Daytime Delay ($25,200\text{s} < ds \le 50,000$s)**:
   The flight departed during the same operational day. Solved strictly on `forward_inner`:
   $$\hat{y}_i = -1.08 + 0.0000 \cdot \hat{y}_{\text{base}, i} + 1.0000 \cdot \text{delta\_sched\_sec}_i$$
   The regression converges to the exact identity.
2. **Extreme Overnight Delay ($ds > 50,000$s)**:
   The flight was pushed across midnight into the next calendar day slot. Solved strictly on `forward_inner`:
   $$\hat{y}_i = 25929.96 - 0.5329 \cdot \hat{y}_{\text{base}, i} + 1.1823 \cdot \text{delta\_sched\_sec}_i$$
   This accounts for the 24-hour cycle offset ($86,400$ s) while maintaining stability.

---

## 4. Verification & Submission v9

- **Protocol Gate Compliance**:
  - `weights_are_val_fitted: false` (all parameters solved on `forward_inner`).
  - Feature fingerprint: `aeade7fee3099c1a` (100% frozen).
  - Pool equality: `d16701419a0d1c8703890a73c854ce9c` ($n=1,387,105$).
- **Submission Binary**: `submissions/PLACEHOLDER_TEAM_v9.parquet`
  - Exactly 215,876 movements scored.
  - 0 nulls, positive bounds $[61, 64409]$ s.
