# Tail diagnosis of the current best model (v31 analogue) on the board's months — result

**Read-only, no training, nothing selected.** `diagnose.py` rebuilds the v31
analogue on seasonal_full TEST (Jan + Jul 2025) from saved predictions: the
3-member FTT/tree/MLP6 blend with forward_inner-solved frozen weights, plus
TabNet's LIRF override at w = 0.55. The 3-member reconstruction matches
`tests_ftt/confirm_seasonal_full.json` exactly (347.9657). The v31 analogue is
pooled 336.72, in-band 251.98.

Outputs: `diagnosis.txt`, `diagnosis.json`, `artifact_rows.txt`, `lirf_rule_check.txt`.

## Where the squared error is

| band (true y) | rows | share of rows | **share of squared error** | v31 RMSE | v31 mean error |
|---|---:|---:|---:|---:|---:|
| in-band ≤ 7,200 s | 344,187 | 99.93 % | 55.9 % | 252 | −1 |
| 7.2k–25.2k | 129 | 0.04 % | 5.6 % | 4,124 | −2,989 |
| 25.2k–80k | 13 | 0.004 % | 10.2 % | 17,513 | −2,804 |
| **≥ 80k ("date artifact")** | **7** | **0.002 %** | **28.2 %** | 39,659 | −32,669 |

- **One LFPG row alone is 17.8 % of all squared error:** y = 84,240, no flight
  record, `delta_sched_sec` = 1,740. **Nothing in its features points to a day's
  delay.** Every component misses it by ~83,400 s. It is unreachable.
- The other 6 artifact rows are LIRF (5 without a flight record), with y ≈ 87,000
  (one day plus taxi) while `delta_sched_sec` ranges 58k–94k. The same feature
  values also appear on LIRF rows whose y **equals** `delta_sched_sec` (e.g. ds
  56,636 → y 56,633; ds 64,438 → y 64,437). A feature-only rule cannot tell
  "identity" from "one day" in that range.
- LIRF in-band is the largest *in-band* slice (7.7 % of rows, 17.0 % of SE, RMSE
  502). The MLP carries it; the tree is far worse there (630), which is why its
  LIRF weight is 0.

## The one feature-only rule on record, re-checked here

`LIRF & no flight record & delta_sched_sec > T → predict delta_sched_sec`:

| T | rows touched | identity holds | pooled 336.72 → |
|---|---:|---:|---:|
| 7,200 | 181 | 46 % | **+7.95** |
| 25,200 (the shipped `idband` rule) | 17 | 71 % | **−12.50** |
| 50,000 | 7 | 29 % | −11.09 |

At T = 25,200 this is exactly `idband`, and locally it looks large. **The board
has already rejected it twice:** v15 (v14 + idband) scored +0.86 worse, and v17
(v16 + idband) +0.86 worse (`board_scores.csv`). The 2025 identity evidently does
not hold on the 2026 scored rows, or the rule's false positives there cost more
than its hits gain. The board is the final authority, and this diagnosis gives
no reason to re-open it. **The lower threshold is actively harmful even locally**
(+7.95 at T = 7,200), which confirms the `TAXI_OUT_DRIVERS` decision not to adopt
the 7.2k–25.2k band.

## Verdict: the tail is closed for the current model

- **38 % of the squared error sits in 20 rows (≥ 25.2k).** Of those, the
  reachable part is exactly the `idband` population the board already rejected,
  and the largest single row (LFPG) has no signal at all.
- The 7.2k–25.2k band (5.6 %) has no identity (46 % at best) and is biased low by
  ~3,000 s in every component. Shifting it would need a tail probability the
  hurdle work showed cannot be estimated at a 0.02 % base rate.
- **The addressable mass is in-band (55.9 %)**, where the gains have always come
  from model quality and averaging (seeds, TabNet, FTT), not from tail rules.

No pre-registered tail experiment is proposed from this diagnosis.
