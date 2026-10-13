# Poisson-objective experiment — result

**Question.** `tests_log_target/RESULT.md` closed log1p+L2 as a clear negative:
it improved val in-band by de-emphasizing the heavy right tail during training,
but lost 22–39s on pooled/test RMSE because the metric that counts charges
`(y-ŷ)²` on rows up to 86,400s, and an unbiased log-space fit is a
downward-biased raw-space fit after `expm1` (Jensen's gap). Does LightGBM's
built-in `poisson` objective avoid that failure? It fits `E[y|x]` directly
(log-link mean, no transform to invert — no Jensen bias) and its score-space
gradient (`μ − y`) weights every row uniformly, unlike Gamma/Tweedie
(`μ^(1−p)`, p>1) which explicitly down-weight large-value rows. Both of those
properties looked like they should place it outside the closed family.

**Setup.** `forward / with_aobt / keep_all`, seeds 42/43/44, both `lightgbm`
(pooled) and `per_airport`. One script (`run_poisson.py`) trains the L2
control and the poisson variant with identical data loading, identical base
hyperparameters, identical early-stopping (patience 200, cap 5000), and
identical `[60, 86400]` output clipping — no transform/inverse-transform
needed for poisson since LightGBM applies the log-link inverse automatically
at `.predict()`. The L2 control reproduces `tests_log_target`'s registered
numbers to the decimal (lightgbm val 297.78, per_airport val 278.17 at
seed42), confirming the two harnesses agree and this comparison isn't
confounded by code drift.

## Result — poisson is a clear negative, same shape as log1p

3-seed means (`summary.txt` / `summary.csv`; rows also foldable into
`results.csv` via `harvest_results.py`, tags `l2ctl` / `poisson`):

| arm | metric | L2 | poisson | Δ | L2 seed-sd | verdict |
|---|---|---:|---:|---:|---:|---|
| lightgbm | val RMSE | 296.68 | 288.05 | **−8.63** | 0.91 | better (decisive) |
| lightgbm | val in-band | 249.86 | 236.52 | **−13.34** | 0.11 | better (decisive) |
| lightgbm | test RMSE | 315.02 | 318.46 | **+3.44** | 1.09 | worse (decisive) |
| lightgbm | test in-band | 229.32 | 226.47 | **−2.85** | 0.64 | better (decisive) |
| lightgbm | test LB-weighted | 287.62 | 290.93 | **+3.31** | 1.08 | worse (decisive) |
| per_airport | val RMSE | 277.86 | 288.73 | **+10.87** | 0.22 | worse (decisive) |
| per_airport | val in-band | 236.76 | 230.36 | **−6.39** | 0.56 | better (decisive) |
| per_airport | test RMSE | 288.19 | 303.69 | **+15.49** | 0.06 | worse (decisive) |
| per_airport | test in-band | 221.92 | 220.97 | −0.95 | 0.21 | better (decisive) |
| per_airport | test LB-weighted | 265.88 | 278.22 | **+12.34** | 0.05 | worse (decisive) |

**Pre-registered bar (decided before running): fails on both arms.** Pooled
test RMSE is decisively *worse* on both lightgbm (+3.44s) and per_airport
(+15.49s) — the bar required improvement, not just in-band improvement.

## Reading

1. **In-band improves, pooled/tail worsens — exactly the log1p signature,**
   just milder in magnitude (log1p: +22 to +39s pooled test; poisson: +3.4 to
   +15.5s). This is now the third time this shape has appeared (log1p val vs.
   test, and now poisson val vs. test on both arms) and is worth naming as a
   diagnostic: **a lever that improves in-band while worsening pooled is
   de-weighting the tail, and on this leaderboard's plain-RMSE metric that
   trade is always a net loss.**
2. **The "uniform gradient weighting" reasoning that motivated this test was
   wrong — LightGBM's tail de-weighting for Poisson happens through the
   Hessian, not the gradient.** Poisson's score-space gradient is `μ − y`
   (uniform), but its Hessian is `h = μ`, and LightGBM's leaf values are
   Newton steps, `≈ −Σg / (Σh + λ)`. A row with a large predicted mean carries
   proportionally more Hessian, so its residual moves the leaf *less* per unit
   of gradient — tail de-weighting through the denominator of the Newton step,
   not the numerator. This is the same underlying mechanism as Gamma/Tweedie
   (`h ∝ μ^p`), just at the mild end (p=1) instead of the aggressive end
   (p=2). The `best_iteration` counts corroborate this independently: poisson
   needs 2–3× the boosting rounds to converge (lightgbm 579–638 vs. L2's
   211–283; per_airport per-airport boosters mostly 250–360 vs. L2's
   100–190) — consistent with damped per-row Newton steps needing more
   rounds to reach the same fit.
3. **This closes the entire objective-function axis by inference, not just
   the tested point.** Four points now measured, ordered by severity of tail
   de-weighting: L2 (best, no de-weighting) → Poisson (+3.4 to +15.5s, mild,
   p=1) → log1p (+22 to +39s, a label transform with its own added Jensen
   bias) → Huber (+82.2s, aggressive residual capping, see
   `TAXI_OUT_DRIVERS.md`'s Huber closure). **Gamma and Tweedie (p between 1.5
   and 2) sit strictly between Poisson and Huber's severity on this axis and
   are closed by inference — do not run them.** Any objective or transform
   that reduces large-residual influence during training will lose pooled
   RMSE on this metric, regardless of the specific mechanism (capping,
   compressing, or Hessian-scaling).

## Provenance note — run-dir tag collision with `tests_log_target`

`run_poisson.py` reused the `l2ctl` tag at the same protocol/seed as
`tests_log_target/run_log_target.py`'s own L2 control, which shares the same
`run_dir(...)` path (arm + protocol + seed + tag). This script ran second and
its control runs **overwrote** `tests_log_target`'s `l2ctl` run dirs' contents
(`model.json`'s `note` field now points at this script). No analytical impact:
`harvest_results.py` only reads `selection_surface`/`val_fitted` from
`model.json`, not the note or script provenance, and both scripts compute the
identical L2 config deterministically from the same seed/data — the eval
numbers are byte-identical to what `tests_log_target/summary.csv` already has
on record, so nothing was lost. Not re-run (would just reproduce the same
numbers at ~33 min of compute). Future scripts reusing a canonical L2 control
tag at an existing protocol/seed should use a distinguishing suffix (e.g.
`l2ctl_pois`) to avoid this.

## Disposition

**Not adopted.** Do not ship a poisson-objective tree arm. Do not run
Gamma or Tweedie — closed by inference above. `tests_log_target/RESULT.md`'s
two suggested follow-ups are also both already closed elsewhere: tuned Huber
(TAXI_OUT_DRIVERS.md, +82.2s, catastrophic) and a routed tail/floor+delta
model (§2q, three independent approaches, all failed to beat plain MLP
out-of-sample on the AOBT-unmatched population). **The objective-function
axis is now fully closed** — L2 remains the shipped choice, and there is no
further variant on this axis worth testing.
