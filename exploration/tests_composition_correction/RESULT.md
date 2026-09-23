# Every selection in this build was made on a biased estimator. Does it matter?

Pure re-analysis — no training, no new runs. All numbers already existed in the
eval JSONs.

## The bias

`common/evaluate.py` has computed `leaderboard_weighted_rmse` for every arm since
the audit stage: pooled RMSE with each airport re-weighted to its share of the
**scored** set, `sqrt(Σ_g w_g · MSE_g)`. Since pooled RMSE over the scored set is
exactly that sum with the scored shares, **the corrected number is the unbiased
estimator of the competition score and pooled val RMSE is a biased one.**

It was never harvested into `results.csv`, and `compare.py:210` ranks on
`val_rmse`. So every arm in this build was selected on the biased estimator.

The two compositions are not close (val vs scored share):

| airport | val | scored | ratio |
|---|---:|---:|---:|
| EGLL | 11.04 % | 18.45 % | **1.67** |
| EHAM | 11.70 % | 17.69 % | **1.51** |
| EDDF | 11.43 % | 16.82 % | **1.47** |
| LTFM | 13.03 % | 10.37 % | 0.80 |
| LFPG | 11.23 % | 8.40 % | 0.75 |
| LEMD | 9.89 % | 7.78 % | 0.79 |
| LSZH | 6.62 % | 4.63 % | 0.70 |
| LIRF | 7.79 % | 5.12 % | **0.66** |
| LEBL | 8.78 % | 5.65 % | 0.64 |
| EDDM | 8.50 % | 5.07 % | 0.60 |

The three largest leaderboard airports are ~1.5× **under**-represented in val;
**LIRF — the airport this build has spent most of its effort on — is 1.5×
over-represented.**

## Does the bias change any decision? No.

Re-ranked all 16 canonical arms on corrected val:

| | pooled val | corrected val |
|---|---|---|
| #1 | `airport_operator_multimodel` | `airport_operator_multimodel` |
| #2/#3 | cross_blend / multimodel_antigravity | **swap** (margin 0.28 s) |
| #7/#8 | per_operator / per_airport | **swap** (margin 0.19 s) |
| everything else | — | unchanged |

**The winner is unchanged and the only two swaps are between arms already
recorded as statistical ties**, on margins of 0.19–0.28 s against seed SDs of
0.27–5.67 s. By this build's own rule those are not rankings.

The level shift is large (15–38 s) but among the top contenders it is nearly
uniform (~16–17 s), which is why the ordering barely moves.

**Finding: the estimator was biased in level, not in ranking. No past selection
needs revisiting.** `compare.py`'s default is therefore left alone — changing
`config.METRIC` and the declared selection set on evidence this weak would be the
same move as the ladder in `overtuned/`. Both columns are now harvested
(`val_lb`, `test_lb`) so the question stays visible.

## Where it does bite: v14

The msmlp gain, decomposed per airport (val, mean of seeds 42/43/44):

| airport | scored w | base | msmlp | delta |
|---|---:|---:|---:|---:|
| **LIRF** | 0.0512 | 551.76 | 538.34 | **−13.42** |
| EDDM | 0.0507 | 174.84 | 174.33 | −0.51 |
| LFPG | 0.0840 | 241.63 | 241.28 | −0.36 |
| LSZH | 0.0463 | 158.93 | 159.04 | +0.11 |
| EGLL · EHAM · EDDF · LTFM · LEMD · LEBL | — | — | — | −0.10 … +0.03 |

**96 % of the total weighted improvement is LIRF.** Seven of ten airports improve,
but every non-LIRF move is ≤0.51 s and three are slightly worse.

Since LIRF is 1.5× over-weighted in val, the corrected gain is smaller than the
pooled one:

| | pooled val | corrected val |
|---|---:|---:|
| single-draw base | 249.77 | 233.11 |
| msmlp | 247.39 | 231.43 |
| **delta** | **−2.38 s** | **−1.68 s** |

**v14 is still better than v13, by roughly 30 % less than the headline said.** The
variance argument (seed spread 4.090 → 0.020 s) is also LIRF-driven, so it carries
the same discount — the benefit is a LIRF-specific variance reduction whose
leaderboard weight is lower than val implies. The decision does not reverse; its
size does.

## Limits of the correction

- It corrects **airport composition only**. It does not correct the temporal shift
  (val is Sep–Oct; the scored period is later), and it does not correct tail-band
  mix — `stage12` showed the scored set has its own tail composition.
- So `val_lb` is a *better* estimator of the leaderboard than `val_rmse`, not a
  prediction of it. Do not quote it as an expected score.

## What this implies for where effort should go

EGLL + EHAM + EDDF are **53 % of the leaderboard and 34 % of val**. Their val
RMSEs are 237 / 162 / 164 — EGLL is the weakest of the three by ~75 s and carries
the single largest weight. Nothing in this build has targeted it specifically;
the effort went to LIRF, which is worth a third as much and is already handled by
a zero-parameter rule plus a blend weight.

That is the strongest available signal about where the remaining headroom is, and
it comes from re-reading numbers that were already on disk.
