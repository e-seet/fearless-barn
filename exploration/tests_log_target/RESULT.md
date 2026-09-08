# log1p(target) experiment — result

**Question.** The pooled RMSE metric is dominated by a heavy right tail (worst
0.1 % of rows carry ~43 % of SSE). Does training the tree arms on `log1p(y)`
instead of raw seconds — compressing that tail so it stops owning the L2
gradient — improve the result?

**Setup.** `forward / with_aobt / keep_all`, seeds 42/43/44. One script
(`run_log_target.py`) trains the L2 control and the log1p variant with identical
data loading, identical params (the shipped `PARAMS`), identical early-stopping
(patience 200, cap 5000; for log1p the ES metric is val RMSE *in log space*), and
identical output clipping to [60, 86400] s. log1p predictions are `expm1`'d before
clipping. Scoring is `common.evaluate.evaluate` — the same metric block every arm
reports. Control reproduces the registered numbers exactly (lightgbm val 297.78,
per_airport val 278.17), so the harness is sound.

## Result — log1p is a clear negative

3-seed means (`summary.txt` / `summary.csv`; rows also in `results.csv`, tags
`l2ctl` / `log1p`):

| arm | metric | L2 | log1p | Δ | L2 seed-sd | verdict |
|---|---|---:|---:|---:|---:|---|
| lightgbm | val RMSE | 296.68 | 319.71 | **+23.0** | 0.91 | worse (decisive) |
| lightgbm | val in-band | 249.86 | 234.23 | **−15.6** | 0.11 | better (decisive) |
| lightgbm | test RMSE | 315.02 | 346.89 | **+31.9** | 1.09 | worse (decisive) |
| lightgbm | test in-band | 229.32 | 229.92 | +0.6 | 0.64 | worse (within noise) |
| lightgbm | test LB-weighted | 287.62 | 314.24 | **+26.6** | 1.08 | worse (decisive) |
| per_airport | val RMSE | 277.86 | 300.09 | **+22.2** | 0.22 | worse (decisive) |
| per_airport | val in-band | 236.76 | 230.18 | **−6.6** | 0.56 | better (decisive) |
| per_airport | test RMSE | 288.19 | 327.04 | **+38.9** | 0.06 | worse (decisive) |
| per_airport | test in-band | 221.92 | 226.11 | **+4.2** | 0.21 | worse (decisive) |
| per_airport | test LB-weighted | 265.88 | 297.91 | **+32.0** | 0.05 | worse (decisive) |

## Reading

1. **log1p does exactly one thing well: val in-band drops 6–16 s.** That is the
   tail-no-longer-dominating-the-gradient effect, and it is real and consistent
   across seeds. It is also the *only* column where log1p wins.
2. **The val-in-band gain does not generalise.** test in-band is flat (lightgbm,
   +0.6) to *worse* (per_airport, +4.2, and that +4.2 is 20× the arm's 0.2 s
   seed-sd — decisive, not noise). So the log-space model fits the val body
   better in a way that does not hold out one quarter later.
3. **Everything the metric actually rewards gets worse by 20–40 s.** Pooled RMSE
   and LB-weighted RMSE both blow out, on both arms, at every seed. Log-space
   training biases large predictions low (an unbiased error in log space is a
   downward-biased error after `expm1`), and this metric charges (y−ŷ)² on rows
   up to 86,400 s.
4. **The transform and the metric are working against each other.** The tail is
   where this leaderboard is decided; log1p's whole mechanism is to de-weight the
   tail. So it cannot help the score here — confirmed rather than argued.

## Disposition

**Not adopted.** Do not ship a log-target tree arm. The idea that the tail should
not dominate the in-band fit is not wrong, but log1p pays for it on the metric
that counts. If pursued, the next forms to try keep the tail *in* the objective
instead of compressing it out:

- Huber / quantile loss with a tuned δ (seed-swept, in-band + pooled reported).
- A two-model split: L2 in-band model + an explicit tail/`has_flight_record`
  model, routed — not a global transform.

Bar for either: must beat the current winner on **test in-band** by more than the
seed spread across ≥3 seeds, with the negative controls passing. log1p fails that
here (test in-band worse on both arms).
