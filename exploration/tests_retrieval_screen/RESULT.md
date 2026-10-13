# Retrieval (TabR-style) screen — result

**Closed at the screen; the full retrieval build is not worth its epochs.**
No training run, so no compute was spent beyond the diagnostic.

## Setup

`run_retrieval_screen.py`. `forward`/`with_aobt`/`keep_all`. For every val and
test row, the retrieved-neighbour mean is the mean training target of the
finest cell with at least K=32 training rows (back-off from ADEP+stand+runway+
hour+aircraft type down to ADEP+hour). Coverage 99.999%. It is compared to the
SHIPPED BLEND's residual (`airport_operator_multimodel`, tag `msmlp`), not to
`y`: `d = retrieval_mean - blend_pred` versus `resid = y - blend_pred`.
Permutation control: `d` shuffled within airport, 200 draws.

Pre-registered bar (before running): out-of-sample pooled-RMSE gain of at
least ~2s (fit on one surface, applied to the other) AND rho above the
permutation p99.

## Result

| | val | test |
|---|---:|---:|
| rho(d, resid) | -0.014 | -0.191 |
| permutation p99(abs rho) | 0.0073 | 0.0095 |
| in-sample OLS gain | +0.038s | +4.558s |
| out-of-sample gain | -3.351s (test-fit -> val) | +0.741s (val-fit -> test) |

**Fails the bar.** The test-surface correlation is real against the
permutation control, but it is not stable across surfaces: the same
relationship applied out of sample gains +0.74s in one direction and loses
3.35s in the other. The sign is also the opposite of what retrieval would
need: a negative slope means the blend is already too close to the neighbour
mean where the two disagree, i.e. the correction pushes AWAY from retrieval.
The most likely reading is that January-August cell means carry tail rows
and seasonal level that do not persist into Nov-Dec, which is a period shift
the blend's own features already absorb better. Not investigated further
because the screen fails on the out-of-sample criterion regardless.

## Disposition

**TabR-style retrieval not built.** Closes the last architecture candidate
with a distinct mechanism (nonparametric neighbour averaging). The
under-observed-categorical regime was the reason to expect it to help;
what the screen shows is that the neighbour mean adds no stable information
on top of the blend even at 99.999% coverage.
