# Architecture-diversity ensemble (MLP + GRU state model) — result

**Closed, negative.** Mixing the GRU state model into an MLP ensemble does not
beat adding MLP seeds at equal member count. No training; screen only.

## Setup

`run_diversity.py`. Saved `seqrnn` checkpoints (seeds 42-44, best epoch,
tests_seqrnn) and the registered untagged `mlp_antigravity` forward runs
(seeds 42-47). Equal-weight mean of predictions, forward val and test.
Pre-registered bar: B (3 MLP + 3 seqrnn) beats A (6 MLP) by >= 2s on test and
> 0 on val, with the swapped control B' agreeing in sign.

## Result (RMSE)

| ensemble | val | test |
|---|---:|---:|
| C  3 MLP (42-44) | 250.51 | 262.43 |
| D  3 seqrnn (42-44) | 250.06 | 261.42 |
| A  6 MLP | 248.44 | 259.96 |
| B  3 MLP + 3 seqrnn | 248.99 | 260.64 |
| B' 3 MLP (45-47) + 3 seqrnn | 248.79 | 260.09 |

B vs A: val -0.55, test -0.68 (worse). B' vs A: val -0.35, test -0.13 (worse).
**Fails the bar.** The reason is visible in one number: the residual
correlation between the 3-MLP ensemble and the 3-seqrnn ensemble is **0.980**
on both val (0.9799) and test (0.9809). The GRU carries essentially the same
error as the MLP, so a member from it is worth about one more MLP seed and
nothing more (D 261.42 vs C 262.43 is a tie within seed noise).

## Side observation

Seed averaging still moves the needle at small counts: 3 -> 6 MLP seeds is
-2.5s test / -2.1s val here, consistent with the earlier seed curve; this
just shows no plateau-breaking from the GRU.

## Disposition

Not built. Rules out "architecture-diverse members" as a way past the
seed-averaging plateau for this family of models: MLP, GRU-state MLP (r =
0.98) and, from earlier screens, the router and TabNet all reproduce the same
errors or are worse. Seeds 42-44 for the GRU only; a 6-seed GRU would not
change a 0.98 residual correlation.
