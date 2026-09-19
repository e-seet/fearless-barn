# TabNet (deep tabular architecture) — result

**Closed, negative against the correct baseline — and the single promising
seed was noise.** Beats plain LightGBM decisively and consistently; does not
beat the existing entity-embedding MLP once seed variance is accounted for,
and carries roughly 3x the MLP's own seed noise.

## Setup

`run_tabnet.py`. `forward`/`with_aobt`/`keep_all`, seeds 42/43/44. TabNet
(`pytorch-tabnet`, newly installed — not previously used in this repo),
`n_d=n_a=64`, `n_steps=5`, categorical embedding sizes matched to the
existing MLP's own sizing rule (`min(48, max(4, sqrt(card)))`) for a fair
architecture-only comparison, not a capacity-budget difference. Preprocessing
mirrors `pipeline_mlp_antigravity/train.py` exactly (train-only category
vocab with a shared unknown code, train-median-filled + standardized
numerics). GPU (RTX 3080 Ti), ~17-24 epochs per seed before early stopping,
~1000-1400s per seed.

## Result

| seed | val RMSE | test RMSE |
|---|---:|---:|
| 42 | 262.87 | 245.05 |
| 43 | 282.87 | 297.30 |
| 44 | 269.00 | 274.47 |
| **mean (sd)** | **271.58 (10.25)** | **272.27 (26.19)** |

vs. `mlp_antigravity` (same protocol, 3-seed mean/sd from `results.csv`):
val 257.01 (3.52), test 269.78 (8.72).

**Seed 42 alone looked like a real win** (test 245.05 vs MLP's own seed42 of
259.71, a -14.66s delta — large enough to have been worth building on).
**That was noise.** Seeds 43/44 came in decisively worse on both val and
test, and TabNet's own across-seed spread (test sd 26.19) is roughly 3x
larger than the MLP's (8.72) — this is a genuinely higher-variance
architecture on this dataset, not merely an unlucky draw or two. Averaged
properly: TabNet loses to the MLP on val by +14.57s (larger than either
model's own seed-sd, decisive) and is a statistical wash on test (+2.49s,
well inside TabNet's own noise band) but with far less reliability.

**Does beat plain LightGBM decisively and consistently** (val -14.9 to
-34.9s, test -18.5 to -70.7s across all three seeds) — TabNet is a real,
working deep-tabular architecture on this data, it just doesn't clear the
bar the existing MLP already set, and the bar it needs to clear is the MLP,
not the tree baseline (the same lesson as the matched/unmatched router:
picking the right comparator changes the verdict).

## Disposition

**Not built further.** No decorrelation-gate/blend check run — moot, since
the standalone result already fails on the primary comparator (same
reasoning as the router closure: a weaker-or-equal, higher-variance
component isn't worth blend-testing before it at least matches what it
would replace). Closes the "deep tabular architecture" direction for this
specific construction (TabNet with matched capacity to the existing MLP).
Not tested: FT-Transformer or NODE specifically — TabNet was chosen as the
most differentiated inductive bias (sequential sparse feature attention) and
representative of the "materially different deep-tabular architecture"
option; a different deep-tabular variant remains untested in principle, but
given TabNet already underperforms the MLP mean while adding 3x the
variance, and this project's broader finding (per §2v's competitive-
intelligence pass) that no top team publishes methodology suggesting a deep-
tabular edge, further deep-tabular variants are low-prior without a more
specific reason to expect a different outcome.
