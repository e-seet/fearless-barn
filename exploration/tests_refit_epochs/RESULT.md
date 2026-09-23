# How many epochs should the `full` MLP refit train? — result

Bar, rules and decision logic: `PREREGISTERED.md`, fixed before any run.
Output: `analysis.txt`, `analysis.json`. 12 seeds (42–53), 3 stages each, all
converged well inside the 50-epoch curves (chosen epochs ≤ 36 < cap 45).

## Pre-registered verdict: ADOPT `median` (the default), **deferred, not executed**

12-seed ensemble mean, in-band RMSE (y ≤ 7200):

| rule | surface 1 (B→C, Nov–Dec) | vs ship | surface 2 (A→B, Sep–Oct) | vs ship |
|---|---:|---:|---:|---:|
| `ship` (10/14 analog) | 233.78 | — | 228.84 | — |
| `median` | 228.95 | **−4.83** | 229.39 | +0.55 |
| `div` (fewer epochs) | 234.73 | +0.95 | 232.39 | **+3.55** |
| `mul` (more epochs) | 225.35 | −8.42 | 228.02 | −0.83 |
| `own` (each seed's own E) | 232.36 | −1.41 | 228.90 | +0.06 |

- `median` is not worse than `ship` by > 0.5 s on surface 1, so the default
  applies. In production that means ceil(12.5) = **13 epochs for every `full` member**.
- **`div` (fewer epochs for more data) is refuted on both surfaces.** The
  update-matched argument in `CONVERGENCE_AUDIT.md` §2 was the wrong direction.
- `mul` beats `median` by ≥ 1.0 on both surfaces but beats `ship` by only 0.83 on
  surface 2, so it narrowly fails the directional bar.

**Why the 13-epoch retrain is deferred.** Moving 10/14 to 13 stays in the region
the curves below say is wrong. Surface 1's −4.83 for `median` also sits on a
jagged stretch of the curve (epochs 13 → 15 → 16 read 234.7 → 229.2 → 227.9),
and surface 2 has `median` +0.55 worse. **13 epochs is the fallback if the
confirmation below fails**, not something to spend a retrain and a slot on now.

## Descriptive finding (post-hoc, NOT adoptable from these data)

**A 12-seed average keeps improving long after single-seed early stopping
would stop.** Uniform-E ensemble in-band RMSE:

| E | 10 | 15 | 20 | 25 | 30 | 35 | 40 | oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| surface 1 (Nov–Dec) | 234.2 | 229.2 | 228.0 | 223.1 | 226.4 | 222.4 | 221.6 | E=49, 221.3 |
| surface 2 (Sep–Oct) | 234.1 | 236.5 | 233.3 | 231.6 | 228.3 | 229.0 | 227.9 | E=41, 227.8 |

Per-seed early stopping picks E on a *single* seed's curve (medians 16.5 and
24). Averaging removes the variance that makes a single seed's later epochs look
worse, so the ensemble prefers **roughly 2–3× more epochs**. That is ~12 s
in-band on surface 1 and ~6 s on surface 2 against production-like 10–16.
Production ships an 18-seed average trained at 10/14 epochs.

**Why it cannot be adopted here:** both target curves have now been seen, so any
rule scored on these surfaces is selected on them. It needs a fresh surface; see
`PREREGISTERED_seasonal.md`.

## Refit sanity (descriptive)

Nov–Dec, each seed at its B-chosen epoch: B (trained Jan–Aug) 239.39 vs C
(trained Jan–Oct) 232.36. **Adding the two most recent months helps by 7 s**,
which supports the full-refit decision itself.

## Fidelity note

The fast GPU loop picks different epochs from the CPU pipeline for the same
seed: B's median is 16.5 against 12.5 for the registered forward runs (seed 42:
13 vs 10). The rules are compared within one consistent system, but a
production change must be trained through the production code path.
