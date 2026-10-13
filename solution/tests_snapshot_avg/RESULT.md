# Snapshot / weight averaging inside one MLP run — result

**Closed, negative.** Averaging epochs (or weights) within a run does not beat
the val-selected best epoch, and does not add anything once seeds are averaged.

## Setup

`run_snapshot.py`, `analyze.py`. Same model/data/hyperparameters as
`pipeline_mlp_antigravity` (forward / with_aobt / keep_all, seeds 42/43/44),
re-implemented with direct GPU minibatch indexing (~5s/epoch vs ~135s), fixed
epochs, val+test predictions stored every epoch. Reproduction check: the
`best` variant's val is 250.92 / 259.35 / 260.98 (mean 257.08) against the
registered arm's 253.99 / 256.16 / 260.88 (mean 257.01); test mean 271.37 vs
269.78. Same distribution, so the re-implementation is a fair stand-in.

## Convergence (asked for explicitly)

- **First attempt, 25 epochs: NOT converged** (seed 43 val still falling, drift
  +24s between early and late window; seed 44 also failed). Its contrasts are
  discarded (`analysis_e25_NOT_CONVERGED.txt`).
- **Second attempt, 60 epochs: passes the gate** (best epoch not pinned at the
  cap; window slope ~ +0.1 to +0.7 s/epoch). Limitation of the gate: it only
  catches "still improving". The curves show val bottoms out around epochs
  13-25 and then RISES (seed 44: late window 10.9s worse than early), i.e.
  converged then overfit. So the pre-registered window (41-60) sits in the
  overfit regime.

## Pre-registered result (window 41-60): fails

Test paired gain vs `best`, per seed / mean: window-mean -10.5, +1.3, -1.5 /
-3.6; last-5 -6.7, +0.1, -8.5 / -5.0; SWA (BN recomputed) -9.9, +2.8, +0.1 /
-2.3. 3-seed ensemble on test: -3.0 / -4.1 / -1.3 (worse than ensemble of
`best`). Val is worse by 9-12s everywhere (the baseline is val-selected, so
val is biased its way; test is the honest column). Bar was >= 3s single-model
in all 3 seeds and >= 2s at ensemble level.

## Exploratory pass (post-hoc, NOT pre-registered): windows near the plateau

From stored predictions, no retraining (`analysis_e60_exploratory.txt`):
- windows centred on the val-selected epoch (same selection information as the
  baseline): test -7 to -8s single, -7 to -9s ensemble. Worse.
- fixed windows: best case epochs 21-30, test +1.05s single (per seed +2.0,
  -6.8, +7.9) and +1.20s ensemble; val -1.2s. Sign-inconsistent across seeds
  and far below the bar. Early windows (11-20) are much worse (-10s).
Nothing here is worth a confirmation run.

## Why it doesn't work

Epochs within one run at constant LR are highly correlated (same trajectory),
so averaging them removes little of the seed-to-seed variance that matters:
test sd across seeds is ~15s, while averaging 3 seeds alone takes test from a
271.4 single-model mean to 262.9 (-8.5s). Snapshot averaging moves the within-run
component only, and pays for it by mixing in worse epochs.

## Disposition

Not built into the pipeline; no arm changes. The local-vs-board lead (board
pays more for MLP variance reduction than local val shows) still stands, but
this cheap within-run route does not deliver it; only across-seed averaging
does, and that has plateaued at 12-18 seeds.
