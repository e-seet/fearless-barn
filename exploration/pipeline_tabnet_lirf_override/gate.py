"""The TabNet seed health gate, as code rather than prose.

This is the gate `tests_tabnet/build_candidate.py` applied when v28/v29 were
built. It lived only inside that script, so the *selection* behind the best
board score (seeds 42/43/45/46/48 used, 47 rejected) was not reachable from
`submit.py` and not re-derivable without re-reading the script. It is lifted
here unchanged so the arm and the historical candidate builder agree by
construction.

Why each condition exists (all three were earned, none are decoration):

* ``val_rmse <= 260`` -- final seed 42 once collapsed to a Dec val of 482 with
  its train loss stuck around 265k. Validated TabNet Dec val is 233-241, so a
  run far outside that band is a failed optimisation, not a weak draw.
* ``best_epoch >= 10`` -- the same collapse signature: a best epoch of 2-3 means
  the run never learned anything and the "best" checkpoint is noise.
* ``not hit_cap`` -- a run that ends because it ran out of epochs has no
  CONFIRMED best epoch, only a last one.

  Read this one as **unresolved, not failed**. Measured 2026-09-21: final seeds
  47 and 51 were rejected here at cap 60 (best epochs 50 and 47); re-run at cap
  75 they early-stopped at 66 and 63 with the *same* best epochs and
  *byte-identical* predictions. They had already found their optimum -- the cap
  only stopped patience-15 from confirming it. The gate was right to be
  cautious (at cap 60, with best at 50, epochs 61-65 were genuinely unobserved),
  but the correct response to a rejection here is to RAISE THE CAP AND RE-RUN,
  which resolves it, not to discard the seed. See ``../CONVERGENCE_AUDIT.md``
  sec.3.
* ``labels_aligned`` -- the marker written after the X/y misalignment bug of
  2026-09-20 (unfiltered y against an outlier-filtered X shifted every label
  after the first dropped row). Seeds saved before that fix do not carry it.
"""
from __future__ import annotations

GATE_VAL_RMSE = 260.0
GATE_MIN_BEST_EPOCH = 10


def seed_health(z) -> "tuple[bool, str]":
    """(healthy, reason). `z` is a loaded ``final_seed<N>.npz``."""
    if "labels_aligned" not in z.files:
        return False, "no labels_aligned marker (saved before the X/y alignment fix)"
    val = float(z["val_rmse"])
    ep = int(z["best_epoch"])
    capped = bool(z["hit_cap"])
    if val > GATE_VAL_RMSE:
        return False, f"Dec val {val:.1f} > gate {GATE_VAL_RMSE}"
    if ep < GATE_MIN_BEST_EPOCH:
        return False, f"best epoch {ep} < gate {GATE_MIN_BEST_EPOCH}"
    if capped:
        return False, f"hit the epoch cap (best epoch {ep} of {int(z['epochs_run'])} run)"
    return True, f"Dec val {val:.2f}, best epoch {ep}, ran {int(z['epochs_run'])}"
