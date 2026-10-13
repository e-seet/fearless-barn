# Ensemble-level epoch count — confirmation on a fresh surface (pre-registration)

Written 2026-09-23, **before any seasonal_full curve was trained**. Follows
`RESULT.md`, whose descriptive finding is that a 12-seed average keeps improving
to ~2–3× the per-seed early-stopped epoch count. Both surfaces that showed it
have been seen, so the finding is post-hoc. This is its test on a surface where
nothing has been looked at.

## Surface

`seasonal_full`: train Feb–Jun + Aug–Nov 2025, val **Dec**, test **Jan + Jul**,
the months the board scores. Same fast-loop recipe as `run_chain.py`, seeds
42–53, a fixed **60 epochs** (the prior oracle optima reached 41–49, so 60 keeps
the rolling window inside the curve), predictions saved at every epoch on Dec
and on Jan+Jul.

## Rules (all read off the saved curves; paired by seed)

| rule | definition |
|---|---|
| `own` | each seed at its own production early-stopped epoch on Dec (patience 6, cap 45), the comparator of how members are chosen today |
| `ship` | seeds 42–51 at seed 42's `own` epoch, 52–53 at seed 44's (the production 10/14 analog) |
| **`ens`** | one uniform E for every seed: the argmin over E ∈ [1, 60] of the **5-epoch centred rolling mean** of the 12-seed-ensemble **Dec** in-band RMSE. The rolling mean guards against the jagged single-epoch minima seen before |

`ens` selects on Dec and is scored on Jan+Jul, which it never sees.

## Bar (all must hold; metrics on Jan+Jul, 12-seed ensemble)

1. **In-band RMSE:** `ens` beats both `own` and `ship` by **≥ 2.0 s**.
2. **LIRF guard:** `ens` LIRF in-band is not worse than `own` by more than **2.0 s**.
   In the forward blend sizing (`blend_sizing.txt`), LIRF got 1–4 s *worse*
   with more epochs, and the board weights LIRF ~2× its local share.
3. **Tail guard:** `ens` pooled RMSE is not worse than `own` by more than **2.0 s**,
   and its count of predictions > 7,200 s is at most 1.5× `own`'s. Max
   prediction reported. Tail predictions are load-bearing on the board (v18
   capped them and scored 491).

**Descriptive only:** the same comparison inside the tree+MLP blend, using
per-airport alphas solved on forward_inner (stage A curves) at the matching E
and the cached seasonal_full tree predictions.

## What a PASS licenses, and what it doesn't

A PASS licenses **proposing** a production chain to the user. It is not
permission to run one. The chain would be:
1. `full` MLP members trained at the `ens`-rule count through the production
   code path. The fast loop and the CPU pipeline pick different epochs for the
   same seed, so the count must be re-derived there, or scaled by the measured
   ratio of their chosen epochs.
2. **Every downstream weight re-solved against members trained that way:**
   - the per-airport tree/MLP alpha (forward_inner MLPs at the matching count);
   - FTT's 3-member weights;
   - TabNet's LIRF w, whose base is the MLP at LIRF.
3. A new tag throughout; v25–v31 must stay rebuildable.

That is several GPU hours plus a submission slot, and it needs the user's go-ahead.

**A FAIL** closes the ensemble-epoch idea. The pre-registered fallback from
`RESULT.md` then stands: a uniform 13 epochs, still deferred until there's a
reason to spend a slot on it.
