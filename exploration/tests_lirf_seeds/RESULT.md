# More MLP seeds for LIRF — result

**Closed, negative.** LIRF's seed-averaging curve plateaus at 6-12 seeds, exactly
like the pooled curve. More seeds buy nothing at LIRF.

## Setup

`train_seeds.py`, `curve.txt`. 24 extra seeds (48-71) of the mlp_antigravity
recipe (patience 6, best-epoch checkpoint, max 25 epochs; fast GPU loop
verified equivalent in tests_snapshot_avg). Curve = mean over 300 random
k-subsets of the 24 seeds (removes ordering luck). Motivation:
tests_local_vs_board/per_airport_split.txt (arms differ only at LIRF; board
~2.0x local LIRF RMSE), and the shipped blend gives LIRF 100% MLP weight, so
the MLP-ensemble LIRF RMSE is the blend's LIRF RMSE.

Convergence: every seed early-stopped or plateaued well before the cap (best
epochs min/median/max 11/14/20 of 25; 19/24 stopped early, none pinned at the
cap; single-seed val mean 256.18, sd 2.65).

## Result (forward; LIRF n=28,809 val / 23,396 test)

| seeds k | LIRF val | LIRF test | non-LIRF mix val | pooled val |
|---:|---:|---:|---:|---:|
| 1 | 556.2 | 507.0 | 210.74 | 256.32 |
| 3 | 540.1 | 495.5 | 206.04 | 250.05 |
| 6 | 536.5 | 491.8 | 204.86 | 248.57 |
| 12 | 535.8 | 490.1 | 204.15 | 247.93 |
| 18 | 534.9 | 489.7 | 203.97 | 247.62 |
| 24 | 534.5 | 489.3 | 203.90 | 247.50 |

6 -> 24 seeds: LIRF -2.0 val / -2.5 test. Essentially all of the LIRF gain is
in the first 3-6 seeds. Using the fitted ~0.33 board-seconds per LIRF-local-
second, 12 -> 18 seeds is worth ~0.3 board-seconds in total, in line with the
actual v24 -> v25 board move of -0.54.

## Disposition

Not built. The shipped 18-seed MLP is already past LIRF's plateau; nothing is
left to gain from seed count at LIRF. (Earlier 6-seed registered curve said the
same: 542.5 -> 536.7 val.)
