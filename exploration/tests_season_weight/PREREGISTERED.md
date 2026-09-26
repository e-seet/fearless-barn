# Season-matched sample weighting, one MLP per scored month — pre-registration

Written 2026-09-26, **before any code was run**. v35 = 287.9247 is best and last
(v36, the C2 probe, may replace it). Ranking uses the last submission.

## Idea, and why it is not the closed recency lever

Plain recency weighting (`lightgbm_recency`, trees) was noise (+0.31 s against a
1.67 s spread) and was ruled out in plan 13 §4b. The trees contribute little
marginally, and weighting toward late 2025 "optimises January, pessimises July".
This differs on both counts:
- **the MLP**, which carries 100 % of LIRF and ~half the blend elsewhere;
- **one model per target month**, each up-weighting the training months nearest
  to it on the calendar circle, so January and July no longer trade off.

Weight of a training row = exp(−d / τ), where d is the circular month distance
between its month and the target month (0–6). **τ = 3, fixed now.** It is mild:
the farthest months keep weight 0.14, so the effective sample stays large, since
the recency work found that collapsing it hurts. The loss is the weighted mean
of squared error within each batch. The model already has `month_num` as a
feature; this changes only what the fit emphasises.

## Screen (fast GPU loop, production MLP recipe; seeds 42–47; cap 50, patience 6)

| surface | base | season-weighted |
|---|---|---|
| **seasonal_full** (train Feb–Jun + Aug–Nov, val Dec, **test Jan + Jul**) | 1 unweighted MLP per seed | 2 MLPs per seed: target Jan (scores the Jan rows) and target Jul (scores the Jul rows) |
| forward (train Jan–Aug, val Sep–Oct, test Nov–Dec) | 1 unweighted MLP per seed | 2 MLPs per seed: target Nov and target Dec |

Early stopping uses each split's val as-is for every model (the same rule for base and
weighted). Metric: in-band RMSE of the 6-seed mean, test part.

## Bar (all must hold)

1. **Jan + Jul in-band improves by ≥ 1.0 s** (weighted − base ≤ −1.0);
2. no single scored month worse (Jan ≤ 0 and Jul ≤ 0);
3. forward Nov–Dec in-band same sign (< 0);
4. **LIRF guard:** Jan + Jul LIRF in-band not worse by more than 2.0 s;
5. every run converged (no cap hit; capped runs are rerun at cap 100 first).

**PASS →** a production proposal to the user: season-weighted MLPs (2 per seed) on
`forward_inner` (target Jul, the weight-solve surface's own months) and `full`
(targets Jan and Jul), blend weights re-solved, TabNet w checked, a new tag, and a
v-candidate shown before `submit.py`. **FAIL →** closed with the recency result.
