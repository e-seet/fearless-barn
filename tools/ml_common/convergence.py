"""Convergence evidence and the sweep-invalidating max_epochs check.

Encodes MISTAKES.md entries #2, #48, #53, #70, and the root CLAUDE.md rule 5.6
("train to convergence, not to a budget").

Convergence means the validation metric has not improved for ``patience`` epochs, or
the learning rate reached its floor with no further gain. ``max_epochs`` is a SAFETY
CAP, not the stopping condition -- hitting it is a signal to investigate, not a result
to accept. The training log must show the evidence: best epoch, stop epoch, patience
counter.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

__all__ = ["convergence_evidence", "assert_sweep_converged"]


def _series(history: Sequence[Any], metric_key: Optional[str]) -> List[float]:
    vals: List[float] = []
    for row in history:
        if isinstance(row, Mapping):
            if metric_key is None:
                raise ValueError("metric_key is required when history rows are mappings")
            if metric_key not in row:
                raise KeyError(
                    "epoch row is missing metric_key {!r}; available keys: {}".format(
                        metric_key, sorted(row.keys())
                    )
                )
            vals.append(float(row[metric_key]))
        else:
            vals.append(float(row))
    return vals


def convergence_evidence(
    history: Sequence[Any],
    *,
    max_epochs: int,
    patience: int,
    metric_key: Optional[str] = None,
    mode: str = "max",
    min_delta: float = 0.0,
    first_epoch: int = 1,
) -> Dict[str, Any]:
    """Derive best_epoch / stop_epoch / patience_counter / hit_max_epochs from a history.

    ``history`` is a list of per-epoch metric values, or a list of per-epoch dicts plus
    a ``metric_key``. ``mode`` is "max" (accuracy, mAP, Dice) or "min" (loss, AbsRel).

    Two traps this function exists to keep you out of:

    * MISTAKES.md #48 -- recompute the best epoch with the SAME criterion the framework
      used to write ``best.pt``, then cross-check it against whatever the trainer
      printed. subfolder94 first read ``argmax(mAP50)`` and got epoch 16, contradicting
      ultralytics' own console line ("Best results observed at epoch 1"); ultralytics
      selects on a composite fitness ``0.1*mAP50 + 0.9*mAP50-95``. A plausible-looking
      best-epoch is exactly the metadata a reviewer trusts without re-deriving.
    * MISTAKES.md #53 -- convergence evidence can look perfect on a schedule that is
      silently 15x shorter than its own parameters describe. subfolder95's inner loop
      ran ``range(steps_per_epoch // batch_size + 1)`` = 13 steps, not 200; the cosine
      LR still hit its floor on cue and early stopping still fired on a real (truncated)
      plateau. Print the ACTUAL batches consumed per epoch once at the start of
      training and assert it matches the intended parameter.

    Returns a dict that is safe to write straight into a training summary artifact.
    """
    vals = _series(history, metric_key)
    if not vals:
        raise ValueError("history is empty -- there is no convergence evidence to derive")
    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'")
    if patience < 0:
        raise ValueError("patience must be >= 0")

    better = (lambda a, b: a > b + min_delta) if mode == "max" else (lambda a, b: a < b - min_delta)

    best_i = 0
    best_val = vals[0]
    for i in range(1, len(vals)):
        if better(vals[i], best_val):
            best_val, best_i = vals[i], i

    n = len(vals)
    stop_epoch = first_epoch + n - 1
    best_epoch = first_epoch + best_i
    patience_counter = (n - 1) - best_i
    hit_max = n >= int(max_epochs)
    early_stop_fired = patience_counter >= int(patience) and not hit_max

    if hit_max:
        note = (
            "HIT max_epochs ({}): the run is truncated, not converged. MISTAKES.md #70 "
            "-- set the cap from an OBSERVED natural best-epoch with at least 1.5x "
            "headroom, never tuned so the first run barely fits. Raise the cap and "
            "re-run into a NEW output directory; do not reuse the truncated one."
        ).format(int(max_epochs))
    elif early_stop_fired:
        note = "early stopping fired at epoch {} after {} epochs without improvement".format(
            stop_epoch, patience_counter
        )
    else:
        note = (
            "run ended before the cap and before patience was exhausted (counter {} of "
            "{}) -- state why it stopped".format(patience_counter, int(patience))
        )

    return {
        "best_epoch": int(best_epoch),
        "best_value": float(best_val),
        "stop_epoch": int(stop_epoch),
        "epochs_run": int(n),
        "patience": int(patience),
        "patience_counter": int(patience_counter),
        "max_epochs": int(max_epochs),
        "hit_max_epochs": bool(hit_max),
        "early_stop_fired": bool(early_stop_fired),
        "converged": bool(early_stop_fired),
        "mode": mode,
        "metric_key": metric_key,
        "note": note,
        "cross_check": (
            "verify this best_epoch against whatever the trainer itself printed or saved "
            "-- if the framework selects on a composite fitness, recompute with THAT "
            "formula (MISTAKES.md #48)"
        ),
    }


def assert_sweep_converged(
    records: Iterable[Mapping[str, Any]],
    *,
    label_key: str = "seed",
) -> Dict[str, Any]:
    """A repeat-seed sweep is INVALID if ANY seed hit ``max_epochs``.

    Encodes MISTAKES.md #70 (subfolder99). ``MAX_EPOCHS = 300`` with ``patience = 30``
    was set after the first run (seed 42) happened to early-stop at 294 -- it just fit.
    The sweep's other two seeds had their last improvement at epochs 290 and 282 and BOTH
    ran to exactly 300 without early stopping ever firing, so each ``best.pt`` came from
    a model that was still climbing. The truncated seeds peaked lower and later than the
    converged one, biasing the 3-seed headline downward; the headline mAP50 column looked
    like a flat plateau, and only the composite fitness the framework actually selects on
    revealed both were still improving.

    Cap-sizing rule: set ``max_epochs`` from the OBSERVED natural best-epoch of a real
    run with at least 1.5x headroom (294 observed -> cap 450), never so the first run
    barely fits. "Any sweep seed hit max_epochs" is a sweep-invalidating condition, not a
    caveat to disclose.

    ``records`` are :func:`convergence_evidence` dicts, one per seed. Raises
    AssertionError naming every offending seed.
    """
    rows = list(records)
    if not rows:
        raise ValueError("no convergence records -- a sweep that reports N seeds must show N")
    truncated = [r for r in rows if r.get("hit_max_epochs")]
    if truncated:
        names = [str(r.get(label_key, "?")) for r in truncated]
        raise AssertionError(
            "repeat-seed sweep INVALID: {} of {} seeds hit max_epochs ({}). A truncated "
            "run is under-trained by definition, and comparing a naturally-converged "
            "seed against truncated ones is not a fair sweep (MISTAKES.md #70). Raise "
            "the cap to at least 1.5x the observed natural best-epoch and re-run those "
            "seeds into NEW output directories.".format(
                len(truncated), len(rows), sorted({int(r["max_epochs"]) for r in truncated})
            )
            + " Offending {}s: {}".format(label_key, names)
        )
    return {
        "gate": "sweep_converged",
        "n_records": len(rows),
        "best_epochs": [int(r["best_epoch"]) for r in rows],
        "stop_epochs": [int(r["stop_epoch"]) for r in rows],
        "all_early_stopped": all(bool(r.get("early_stop_fired")) for r in rows),
        "note": (
            "a build that reports N seeds must be able to show N convergence records "
            "(MISTAKES.md #38 recurrence audit)"
        ),
    }
