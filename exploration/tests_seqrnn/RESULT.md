# GRU sequence-state encoder — result

**Closed, flat — no reliable gain, same verdict as queue-attention despite a
genuinely different construction.** Seed 42 alone looked like a small,
consistent, real win; seeds 43/44 erased it on averaging.

## Setup

`run_seqrnn.py` + `sequence_features.py` + `model.py`. `forward`/`with_aobt`/
`keep_all`, seeds 42/43/44. A GRU (hidden=32) over each departure's backward-
only movement history (K=24 steps, BOTH departures and arrivals at the
airport, oldest to newest, causal only — no future movements, unlike
`pipeline_queueattn_antigravity`'s bidirectional K-nearest window), final
hidden state concatenated into the SAME entity-embedding tabular MLP
architecture as `mlp_antigravity` (byte-identical shape otherwise, so any
delta is attributable to the added sequence branch). GPU (RTX 3080 Ti),
~163s/epoch, 18-24 epochs per seed before early stopping.

## Result

| seed | val RMSE | test RMSE |
|---|---:|---:|
| 42 | 250.74 | 256.19 |
| 43 | 256.27 | 267.27 |
| 44 | 262.09 | 279.65 |
| **mean (sd)** | **256.37 (5.68)** | **267.70 (11.74)** |

vs. `mlp_antigravity` (3-seed mean/sd, `results.csv`): val 257.01 (3.52),
test 269.78 (8.72).

**Mean delta: val -0.64s, test -2.08s — both far inside either arm's own
seed noise.** Statistically indistinguishable from the existing MLP.
Seed 42 alone beat the MLP on both val (-3.25s) and test (-3.52s), in the
same direction on both surfaces (unlike TabNet's discordant single-seed
result, tests_tabnet/RESULT.md) — a more convincing-looking single draw —
but seeds 43/44 both landed worse than the MLP on both surfaces, and the
average washes out to a tie. This is the second architecture this session
where a promising seed-42 draw did not survive averaging (TabNet was the
first, more dramatically); the project's "multi-seed before trusting any
delta" rule caught both.

**Converges to the same empirical conclusion as
`pipeline_queueattn_antigravity`** (closed flat at 6 seeds: mean diff
+0.670s worse, not distinguishable from zero, `TAXI_OUT_DRIVERS.md` §2k) —
despite this being a materially different construction (causal-only GRU
state vs. bidirectional fixed-K attention; includes arrivals, not just
other departures; recurrent processing of a longer window vs. an attention
pool over 8 neighbours). Two independent architectures for "give the model
richer per-flight local traffic structure" both land at the same place:
flat. This is reasonably strong evidence the ceiling here isn't about
attention-vs-recurrence or which neighbours are included — the existing
scalar congestion aggregates (`dep_prev_*m`, `arr_prev_*m`,
`same_rwy_prev_*m`, the congestion rolling mean) already capture most of
what a richer per-neighbour representation of the same underlying movement
stream has to offer.

Does beat plain LightGBM decisively on both val (-35.7 to -47.0s) and test
(-36.1 to -59.6s) across all three seeds — same shape as TabNet: a real
working architecture relative to a tree baseline, just not an improvement
over the arm it would need to replace.

## Disposition

**Not built further.** Closes the sequence/state-model direction (architecture
4 of 4 in this session's "different base architecture" investigation) at the
same empirical bar already established for TabNet and the matched/unmatched
router: beats a weak comparator, ties or loses against the correct one. No
decorrelation/blend check run — moot given the tie on standalone performance
(nothing to blend in that the MLP doesn't already provide). Combined with
queue-attention's prior closure, this reasonably forecloses "richer per-
flight neighbour/sequence representation" as a productive direction for this
pipeline without a fundamentally different data source (not just a
fundamentally different way of consuming the same movement stream).
