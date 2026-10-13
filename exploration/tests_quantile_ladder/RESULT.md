# Quantile-ladder-to-mean — result

**Closed, decisively negative.** Joins Huber/log1p/Poisson as a fourth
member of the objective-function-axis closures — from a genuinely different
construction (fit the full distribution, integrate to the mean) rather than
a modified point-estimate loss — but lands in the same place.

## Setup

`run_quantile_ladder.py`. `forward`/`with_aobt`/`keep_all`, seed 42. 14
independent LightGBM quantile regressors (alpha = 0.01 to 0.999), same base
hyperparameters as the registered `lightgbm` recipe, differing only in
`objective`/`metric`/`alpha`. Per-row quantiles sorted (Chernozhukov
rearrangement) to fix crossing before integrating. Mean recovered by
trapezoidal integration between fitted levels, **with an explicit
exponential-tail extrapolation beyond q=0.999** (fit from the last two
quantiles) rather than truncating the tail at the highest fitted level —
truncation was flagged before running as the failure mode that already sank
the outlier-policy/prediction-cap experiments (config.OUTLIER_POLICIES,
v27 board result), so this was built to avoid rigging the test toward a
known-bad shape.

## Result

| metric | L2 control | q-ladder mean | delta |
|---|---:|---:|---:|
| val RMSE | 297.78 | 331.07 | **+33.29** |
| val in-band | 249.99 | 236.63 | -13.36 |
| test RMSE | 315.79 | 356.08 | **+40.30** |
| test in-band | 229.73 | 230.14 | +0.41 |
| test pred mean | 984.03 | 981.64 | (true mean 995.99) |

**Same shape as every other closed objective-function variant**: improves
the well-behaved in-band bulk (val -13.36s) but loses decisively on pooled
RMSE (+33 to +40s, larger in magnitude than Poisson's already-closed +3.4 to
+15.5s, in the same direction as Huber's +82.2s). The aggregate predicted
mean is close to the true mean (981.64 vs 995.99) — the miscalibration isn't
in the average, it's in which SPECIFIC rows get the large predictions. L2
directly optimizes squared error and is pulled hard toward fitting the
particular high-leverage extreme rows; a per-row quantile-ladder mean, even
with a deliberately tail-aware integration, doesn't reproduce that same
row-specific precision on the extreme tail.

**A real practical defect surfaced along the way, not just the headline
number**: severe per-row quantile crossing before sorting -- 53% of val rows
and 56% of test rows needed reordering. Independently-trained quantile
regressors at nearby alpha levels disagree enough, row by row, that "sort to
fix monotonicity" is doing real work, not a rare edge-case patch. Combined
with the widely varying best_iteration across the ladder (215 to 1109
rounds), this points to the same root cause as the RMSE result: pinball
loss at any single quantile level is a much noisier training signal per row
than direct L2, especially in the sparse extreme-quantile region that
matters most for this heavy-tailed target.

## Disposition

**Not built.** Closes the distributional/quantile-regression direction by
the same standard already applied to Huber/log1p/Poisson
(`TAXI_OUT_DRIVERS.md` sec.2i/2ac): a real, decisive, non-noise pooled-RMSE
regression, in the same direction as every other tail-de-weighting variant.
No multi-seed follow-up run -- the effect size (+33 to +40s) is far past
what 2-3 seeds of noise could plausibly erase, matching this project's
"decide the bar before seeing the number" convention for when a single-seed
screen is already decisive enough to close without further compute.
