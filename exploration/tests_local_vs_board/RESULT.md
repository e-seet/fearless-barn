# Local vs board across all submissions with a matching local recipe

Join of `submissions/submissions.csv` (board score) with `results.csv` (local
`forward` val/test RMSE for the same arm/tag). Spreadsheet join, no training.
The local numbers for v13-v17 and v19 come straight from `results.csv`; v25's
and v26's come from the notes in `TAXI_OUT_DRIVERS.md` sec.2m/2o (v25's 248.33
is the 2-arm blend, NOT the 18-seed recipe -- no forward run of that exists);
v27's 333.0 is from sec.2ai.

| submission | local val | local test | board | board-val | board-test |
|---|---:|---:|---:|---:|---:|
| v2 lightgbm | 291.80 | - | 405.75 | 114.0 | - |
| v3 per_airport_stack | 281.93 | 286.44 | 386.22 | 104.3 | 99.8 |
| v19 airport_operator_blend | 272.88 | 285.80 | 382.92 | 110.0 | 97.1 |
| v27 aob winsor_7200 | 333.00 | - | 496.09 | 163.1 | - |
| v13 1-seed MLP blend | 248.33 | 246.89 | 324.28 | 76.0 | 77.4 |
| v14 3-seed msmlp | 247.39 | 239.87 | 306.32 | 58.9 | 66.4 |
| v15 idband_msmlp | 246.66 | 238.64 | 307.18 | 60.5 | 68.5 |
| v16 congest_msmlp | 247.10 | 237.02 | 306.33 | 59.2 | 69.3 |
| v17 idband_congest | 246.36 | 235.77 | 307.19 | 60.8 | 71.4 |
| v25 (local = 2-arm proxy) | 248.33 | 242.09 | 295.21 | 46.9 | 53.1 |
| v26 3-way catboost | 244.56 | 242.08 | 295.48 | 50.9 | 53.4 |

Fits (board on local):

- all 11, val: board = -245.8 + 2.241 x local, r = 0.990, residual sd 9.4
- 9 with test, test: board = -101.3 + 1.695 x local, r = 0.972, residual sd 8.9
- MLP blends only (7), val: r = 0.46; test: r = 0.34 (not identifiable)
- tree-only (4), val: slope 1.99, r = 0.991

## Reading

1. The local-to-board offset is NOT a constant ~47-51s. It runs +47 to +163s
   and grows with local RMSE (slope ~2.2 on val, ~1.7 on test), so the board
   amplifies local differences.
2. Board-minus-local shrinks as MLP weight/seed count rises (+76 for one
   seed, +59 for three, ~+47-51 for 18 seeds / CatBoost), i.e. the board pays
   more for MLP variance than local val shows. Going from 1 to 3 seeds moved
   local val by -0.9 and the board by -18.
3. The fit is driven by the two clusters (tree-only vs MLP blends); within
   the MLP cluster the residual sd (~9 board-seconds) is larger than any
   local delta available, so it cannot resolve local gains under ~4s. That
   agrees with the ~5s board-visibility floor already on record.
4. Implied local target for a top-team board score: board 245 corresponds to
   local val ~219 on this fit (extrapolated below the observed MLP cluster,
   so treat as a rough guide, not a measurement).
