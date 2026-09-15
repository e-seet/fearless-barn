"""Trivial baselines and MEASURED chance levels.

Encodes MISTAKES.md entries #5, #26, #27, #31, #47.

Two separate things live here and they are not interchangeable:

  * a TRIVIAL BASELINE is a degenerate PREDICTOR (always the majority class, always
    the training mean, detect nothing) scored with the real metric. It answers "would
    a model that learned nothing have done this well?";
  * a MEASURED CHANCE level is the metric's expected value under RANDOM output on this
    exact split, obtained by actually scoring random predictions many times. It is the
    reference a negative control is judged against.

Never derive either from a closed form when you can measure it. MISTAKES.md #27:
subfolder88 emitted ``chance_mAP@10`` as the mean relevant fraction ``R/Ng`` -- the
right chance level for P@k, wrong for mAP@k, because P@i and rel_i are positively
correlated along a ranking. At p = 1/6, k = 10 the true value is 0.070, not 0.167, a
2.4x overstatement; the module's own self-test caught it (0.333 assumed vs 0.191
measured) before any result was computed. The key was DELETED rather than corrected,
and chance is now measured by running a random ranking on the same split.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .splits import present_classes

__all__ = [
    "trivial_baselines",
    "measured_chance",
    "stated_floor",
]


def trivial_baselines(
    y_true: Sequence[Any],
    metric_fn: Callable[[Sequence[Any], Sequence[Any]], float],
    *,
    task: str = "classification",
    seed: int = 0,
    prior: Optional[Sequence[float]] = None,
    classes: Optional[Sequence[Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """The floor every model must clear, computed FROM ``y_true`` and reported beside it.

    ``task="classification"`` produces:
      * ``majority``     -- always predict the most frequent class of ``y_true``;
      * ``prior_sample`` -- sample predictions at the observed class rate.
    ``task="regression"`` produces:
      * ``mean``   -- always predict the mean of ``y_true``;
      * ``median`` -- always predict the median.

    The prior is estimated FROM ``y_true`` by default and every entry records
    ``prior_source``. MISTAKES.md #26 is why that field exists: subfolder86 read
    per-class counters off a screenshot of the INCUMBENT deployed system's HUD and used
    them as the class prior in four places, each labelled "measured" with a prose
    caveat. Those counters are another model's PREDICTIONS -- anything derived from
    them imports that model's biases as facts, and the error is directional, not merely
    noisy: if the incumbent under-detects the hardest minority class, every downstream
    number is wrong in the same direction. A caveat in prose does not remove a numeric
    dependency; only deleting the dependency does. Passing ``prior=`` explicitly is
    supported for a sensitivity check and is recorded as ``prior_source: supplied``.

    Also relevant: MISTAKES.md #5. Under class imbalance, a majority predictor scores
    well above 0.5 on ACCURACY, so a negative control floored at "0.5" can be beaten
    without learning anything. Pass the metric you actually report (macro-F1, AUROC,
    balanced accuracy), not accuracy, and read the floor this function returns for it.
    """
    y = list(y_true)
    if not y:
        raise ValueError("y_true is empty")
    rng = np.random.default_rng(int(seed))
    out: Dict[str, Dict[str, Any]] = {}

    if task == "regression":
        arr = np.asarray(y, dtype=np.float64)
        for name, val in (("mean", float(arr.mean())), ("median", float(np.median(arr)))):
            pred = [val] * len(y)
            out[name] = {
                "metric": float(metric_fn(y, pred)),
                "constant_prediction": val,
                "prior_source": "empirical_from_y_true",
            }
        return out

    if task != "classification":
        raise ValueError("task must be 'classification' or 'regression', got {!r}".format(task))

    if classes is not None:
        labels = list(classes)
    else:
        # Derive from the data via the one shared implementation (MISTAKES #29) --
        # a second local copy of the dedupe/sort logic would drift from splits.py's.
        labels = present_classes(y)
    key_of = {str(v): i for i, v in enumerate(labels)}
    counts = np.zeros(len(labels), dtype=np.float64)
    for v in y:
        counts[key_of[str(v)]] += 1.0

    majority_label = labels[int(counts.argmax())]
    maj_pred = [majority_label] * len(y)
    out["majority"] = {
        "metric": float(metric_fn(y, maj_pred)),
        "predicted_class": majority_label,
        "class_share": {str(labels[i]): float(counts[i] / counts.sum()) for i in range(len(labels))},
        "prior_source": "empirical_from_y_true",
    }

    p = np.asarray(prior, dtype=np.float64) if prior is not None else counts
    total = float(p.sum())
    if total > 0:
        sampled_idx = rng.choice(len(labels), size=len(y), p=p / total)
        sampled = [labels[i] for i in sampled_idx]
        out["prior_sample"] = {
            "metric": float(metric_fn(y, sampled)),
            "prior_source": "supplied" if prior is not None else "empirical_from_y_true",
            "prior": [float(x) for x in (p / total)],
            "warning": (
                None if prior is None else
                "a supplied prior must not come from another deployed system's output "
                "distribution -- labels, class mix and any band calibrated against it "
                "are all excluded from truth (MISTAKES.md #26)"
            ),
        }
    return out


def measured_chance(
    metric_fn: Callable[..., float],
    y_true: Sequence[Any],
    *,
    n_draws: int = 50,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
    sampler: Optional[Callable[[np.random.Generator, int], Any]] = None,
    metric_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """MEASURE the metric's chance level by actually scoring random output. Never assume it.

    Encodes MISTAKES.md #27 (a closed-form chance for a rank-sensitive metric was 2.4x
    too high, and would have waved through a 3.5x-above-chance permuted result), #31
    (a protocol whose val gallery held a single class made mAP@k identically 1.0 -- if
    chance equals the CEILING the split cannot select anything, and the only way to
    find that out is to measure), and #33/#44 (the same discipline applied to a model:
    a pretrained -- or even a randomly initialised -- backbone's null level is measured,
    not assumed).

    By default the random predictor is a permutation of ``y_true``, which is
    rate-matched (it reproduces the class marginal exactly) and therefore cannot be
    beaten by exploiting the base rate the way a fixed 0.5 floor can (#5). Supply
    ``sampler(rng, n) -> predictions`` for anything else: a random ranking / random
    score vector for retrieval, random boxes for detection.

    Returns mean / std / min / max over ``n_draws`` draws. Report the MEAN as the chance
    level and the std beside it -- a chance estimate is a distribution too.
    """
    if n_draws < 5:
        raise ValueError("n_draws must be >= 5; one draw is not an estimate of a null")
    generator = rng if rng is not None else np.random.default_rng(int(seed))
    y = list(y_true)
    if not y:
        raise ValueError("y_true is empty")
    kwargs = dict(metric_kwargs or {})

    draws: List[float] = []
    for _ in range(int(n_draws)):
        if sampler is None:
            order = generator.permutation(len(y))
            pred = [y[i] for i in order]
        else:
            pred = sampler(generator, len(y))
        draws.append(float(metric_fn(y, pred, **kwargs)))

    arr = np.asarray(draws, dtype=np.float64)
    return {
        "chance": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_draws": float(len(arr)),
        "method": "measured by scoring random output on this split, not a closed form",
    }


def stated_floor(metric_name: str, floor_value: float, derivation: str) -> Dict[str, Any]:
    """Record a metric's derived floor next to the number, for metrics with no intuitive one.

    Encodes MISTAKES.md #47 (subfolder94). A classical tracking arm scored test MOTA
    -0.379 and was triaged on believability alone. MOTA with zero detections emitted is
    exactly ``1 - (0 + GT + 0)/GT = 0`` -- a real, DERIVABLE trivial floor. Without it
    stated, a reader sees "MOTA = -0.38" and reads "underperforms", when the correct
    read is "actively worse than emitting no detections at all". MOTA is unbounded
    below, so "0 is the floor" is not self-evident the way "0.5 for balanced binary
    accuracy" is: it has to be derived and written down.

    Use this for MOTA, custom composite scores, any F1 formulation with an unusual
    denominator, and any metric a reader would have to derive the floor for themselves.
    """
    if not derivation.strip():
        raise ValueError("derivation must state HOW the floor was derived, in words")
    return {
        "metric": metric_name,
        "floor": float(floor_value),
        "derivation": derivation,
        "note": "state this beside the headline number (MISTAKES.md #47)",
    }
