# Season-matched per-month MLP weighting — result: FAIL

Bar: `PREREGISTERED.md` (τ = 3, fixed in advance). `screen.py` → `screen.txt` /
`screen.json`. All 36 runs converged (best epochs 10–26, cap 50). Effective sample
76–86 % of n.

In-band RMSE of the 6-seed mean, per-month models vs one unweighted model:

| surface | base | weighted | Δ |
|---|---:|---:|---:|
| **Jan + Jul** | 252.919 | 254.622 | **+1.703** |
| Jan | 219.012 | 220.595 | +1.583 |
| Jul | 277.255 | 279.054 | +1.798 |
| **LIRF (Jan + Jul)** | 481.503 | 492.010 | **+10.508** |
| forward Nov–Dec | 241.069 | 234.981 | −6.088 |
| forward Dec | 256.968 | 246.441 | −10.527 |

Jan + Jul pooled improved (−8.31), but pooled on that surface is dominated by 20
tail rows (`tests_tail_diagnosis`) and is not the bar.

**FAIL** on the decisive conditions: Jan + Jul, per-month, and the LIRF guard.
This is the sixth lead this week to help the forward months and not Jan + Jul.
Up-weighting the calendar-adjacent months helped December, which follows the
training data (Jan–Aug) as the board's months follow 2025. It hurt January and
July 2025, which sit before and between the seasonal_full training months.
**Instrument note:** here the two surfaces disagree strongly (−6.1 against +1.7),
and plan 16 notes that seasonal_full is the less board-shaped of the two. The
pre-registered rule is nonetheless kept. Several leads this week passed forward
and failed on the board, and none has passed forward alone and then succeeded there.
