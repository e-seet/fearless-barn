"""Negative controls -- the experiments that are supposed to FAIL.

Encodes MISTAKES.md entries #5, #14, #22, #23, #27, #33, #34, #44, #49, #51, #52,
#66, and the workspace's standing rule that a control's expected level under the
null must be MEASURED, never assumed.

Four rules govern everything in this module. They are stated once here and cited
from each function's docstring:

  R1  MEASURE THE NULL, DO NOT ASSUME IT. (#5, #27, #33, #44)
      A permuted-label run does not collapse to "chance" unless the initialisation
      itself carries no task signal. An ImageNet-pretrained backbone already scored
      0.86 mAP@10 on subfolder88's task with no labels at all; scored against the
      0.07 measured-chance level, a perfectly clean pipeline looked 13x-above-chance
      leaky. Even a RANDOMLY initialised ResNet-18 measured 0.7965 on that
      texture-driven task (#44) -- so "random init must be near chance" is also an
      assumption, not a fact. Every reference level in this module is either a
      measured chance value, or a measured init_reference, and the API refuses to
      run without one.

  R2  A NULL DISTRIBUTION, NOT A NULL SAMPLE. (#23)
      One permutation on subfolder86's 14-carcass val split scored 0.630 macro-F1
      against a 0.483 chance estimate -- pure sampling noise, read as a structural
      leak. Run n_repeats >= 5 permutations, verdict on the MEAN, and report the
      spread: a spread wider than the tolerance means the control has little
      resolving power on that split, which is itself the finding. Do not widen the
      tolerance until it passes; that disables the control permanently.

  R3  APPLY THE PERTURBATION WHERE THE MODEL CONSUMES IT. (#14)
      subfolder81 permuted entity dicts across records; the BIO alignment step then
      matched nothing and every label became O, so the "permuted" run scored 0.0 for
      ANY architecture -- a tautological collapse testing nothing. If an intermediate
      transformation can absorb, neutralise, or degenerate the perturbation before
      the model sees it, the control is vacuous. Permute at the level the model
      actually consumes.

  R4  A CONTROL MUST BE SHOWN TO FAIL UNDER A BROKEN IMPLEMENTATION. (#22)
      subfolder86's first shuffled-ID tracking control relabelled track t as
      (t+1) % N in every frame and asserted IDF1 < 1.0. IDF1 is (correctly)
      invariant to a consistent global renaming, so the control passed on a BROKEN
      IDF1 too -- a control with no power is worse than no control, because it
      manufactures confidence. Every control here has a paired test asserting it
      FAILS on a deliberately broken input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from .seed import preserve_determinism_flags, seed_everything

__all__ = [
    "ControlResult",
    "ControlSuite",
    "label_permutation_control",
    "overfit_one_batch",
    "overfit_one_batch_generic",
    "select_signal_batch",
    "BEST_CHECKPOINT_KEY",
]

#: Where a best-by-val reading is parked so it can never be mistaken for the verdict
#: (MISTAKES.md #34 / #44).
BEST_CHECKPOINT_KEY = "permuted_val_BEST_checkpoint_do_not_use"


@dataclass
class ControlResult:
    """One control's outcome. ``measured`` holds only numbers, so it is JSON-safe."""

    name: str
    passed: bool
    expectation: str
    measured: Dict[str, float] = field(default_factory=dict)
    detail: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "expectation": self.expectation,
            "measured": dict(self.measured),
            "detail": self.detail,
            "extra": dict(self.extra),
        }

    def summary_line(self) -> str:
        return "{:4s}  {:34s} {}".format(
            "PASS" if self.passed else "FAIL", self.name, self.measured
        )


@dataclass
class ControlSuite:
    """Collects ControlResults so a stage gate can assert on all of them at once."""

    results: List[ControlResult] = field(default_factory=list)

    def add(self, result: ControlResult) -> ControlResult:
        self.results.append(result)
        return result

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failed_names(self) -> List[str]:
        return [r.name for r in self.results if not r.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed_names(),
            "results": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        return "\n".join(r.summary_line() for r in self.results)


# --------------------------------------------------------------------------- #
# 1. Label permutation
# --------------------------------------------------------------------------- #


def _resolve_reference(value: Any) -> Optional[float]:
    if value is None:
        return None
    if callable(value):
        return float(value())
    return float(value)


def _coerce_score(raw: Any) -> Dict[str, Optional[float]]:
    """Normalise a scoring callback's return into converged / best-checkpoint."""
    if isinstance(raw, Mapping):
        if "converged" not in raw:
            raise ValueError(
                "score_permuted returned a mapping without a 'converged' key. "
                "MISTAKES.md #34/#44: the verdict must come from the CONVERGED "
                "(last.pt) weights; a best-by-val reading belongs under "
                "'best_checkpoint' and is reported, never used."
            )
        conv = float(raw["converged"])
        best = raw.get("best_checkpoint", None)
        return {"converged": conv, "best_checkpoint": None if best is None else float(best)}
    return {"converged": float(raw), "best_checkpoint": None}


def label_permutation_control(
    y_train: Sequence[Any],
    score_permuted: Callable[..., Any],
    *,
    measured_chance: Any = None,
    init_reference: Any = None,
    n_repeats: int = 5,
    tolerance: float = 0.05,
    real_arm_score: Optional[float] = None,
    real_arm_margin: Optional[float] = None,
    groups: Optional[Sequence[Any]] = None,
    seed: int = 0,
    name: str = "label_permutation",
    higher_is_better: bool = True,
) -> ControlResult:
    """Shuffle the training labels, refit, and require the metric to fall to the null.

    Rules R1 (measure the null), R2 (n_repeats >= 5, verdict on the mean), R3 (permute
    where the model consumes) and R4 (a broken pipeline must fail this) all apply --
    see the module docstring.

    Parameters
    ----------
    y_train:
        The TRAINING labels, at the level the model consumes them (R3 / #14). If the
        model reads BIO tag sequences, pass tag sequences -- not the source entity
        dicts an alignment step will silently degenerate to all-O.
    score_permuted:
        ``score_permuted(permuted_labels, repeat_index) -> float | mapping``. It must
        train on the permuted labels and return the metric measured on an untouched
        validation set. When a training harness with best-by-val checkpointing is
        involved it MUST return a mapping ``{"converged": x, "best_checkpoint": y}``:
        MISTAKES.md #34/#44 measured 0.8839 from the best checkpoint against 0.1071
        from the converged model on the SAME run. Under a null hypothesis, best-by-val
        selects the epoch that damaged the initialisation least, so it structurally
        reports the initialisation. The verdict here is computed from ``converged``
        only; ``best_checkpoint`` is reported under the key
        ``permuted_val_BEST_checkpoint_do_not_use``.
    measured_chance:
        A MEASURED chance level (float or zero-arg callable) -- e.g. from
        ``baselines.measured_chance``. Never a closed-form prevalence: #27 measured
        mAP@10 chance at 0.070 where the assumed ``R/Ng`` formula said 0.167.
    init_reference:
        The metric of the UNTRAINED but initialised model (float or zero-arg callable).
        Required whenever the arm starts from pretrained weights, and worth measuring
        even for random init (#44). When given, it -- not chance -- is the reference
        the permuted score is judged against.
    real_arm_score:
        The honest arm's metric. When given, the control additionally requires the
        permuted run to sit clearly below it (by ``real_arm_margin``, default
        ``tolerance``), which is what distinguishes "the labels mattered" from "the
        initialisation was already good".
    groups:
        Permute label ASSIGNMENT across groups rather than across items. Shuffling
        item labels inside a group leaves the group's majority label intact and the
        control passes for the wrong reason (subfolder86's carcass-level lesson).
    higher_is_better:
        Set False for loss-like metrics; the comparison direction flips.

    Raises
    ------
    ValueError
        If neither ``measured_chance`` nor ``init_reference`` is supplied. There is no
        default reference level, on purpose (R1).
    """
    chance = _resolve_reference(measured_chance)
    init_ref = _resolve_reference(init_reference)
    if chance is None and init_ref is None:
        raise ValueError(
            "label_permutation_control requires a MEASURED reference: pass "
            "measured_chance=... (see baselines.measured_chance) and/or "
            "init_reference=... (the untrained-but-initialised model's metric). "
            "MISTAKES.md #27/#33/#44: assuming a null level -- a closed-form chance, "
            "or 'a random net must be near chance' -- has flagged clean pipelines as "
            "catastrophically leaking in this workspace three times."
        )
    if n_repeats < 5:
        raise ValueError(
            "n_repeats must be >= 5. MISTAKES.md #23: one permutation is a single "
            "draw from a wide null, not an estimate of it (0.630 vs 0.483 chance on a "
            "14-group split was pure sampling noise, read as a leak)."
        )

    y = list(y_train)
    if not y:
        raise ValueError("y_train is empty")
    groups_list = list(groups) if groups is not None else None
    if groups_list is not None and len(groups_list) != len(y):
        raise ValueError("groups must be the same length as y_train")

    reference = init_ref if init_ref is not None else chance
    reference_source = "init_reference" if init_ref is not None else "measured_chance"

    scores: List[float] = []
    best_ckpt_scores: List[float] = []
    reassigned: List[float] = []

    for r in range(n_repeats):
        rng = np.random.default_rng(int(seed) + 7919 * r)
        if groups_list is None:
            order = rng.permutation(len(y))
            permuted = [y[i] for i in order]
        else:
            uniq = sorted({str(g): g for g in groups_list}.keys())
            # one label per group (the group's first label), then shuffle the mapping
            first_label: Dict[str, Any] = {}
            for g, lab in zip(groups_list, y):
                first_label.setdefault(str(g), lab)
            shuffled_vals = [first_label[k] for k in uniq]
            perm_order = rng.permutation(len(uniq))
            mapping = {uniq[i]: shuffled_vals[j] for i, j in zip(range(len(uniq)), perm_order)}
            permuted = [mapping[str(g)] for g in groups_list]
        changed = float(np.mean([a != b for a, b in zip(y, permuted)]))
        reassigned.append(changed)
        if changed == 0.0:
            raise AssertionError(
                "permutation left every label unchanged -- the control has no power "
                "(MISTAKES.md #22: a control that cannot fail is worse than none)"
            )
        parts = _coerce_score(score_permuted(permuted, r))
        scores.append(parts["converged"])
        if parts["best_checkpoint"] is not None:
            best_ckpt_scores.append(parts["best_checkpoint"])

    mean_score = float(np.mean(scores))
    spread = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    margin = tolerance if real_arm_margin is None else float(real_arm_margin)

    if higher_is_better:
        near_reference = mean_score <= reference + tolerance
        below_real = True if real_arm_score is None else mean_score < float(real_arm_score) - margin
    else:
        near_reference = mean_score >= reference - tolerance
        below_real = True if real_arm_score is None else mean_score > float(real_arm_score) + margin

    passed = bool(near_reference and below_real)
    weak = spread > tolerance

    measured: Dict[str, float] = {
        "mean_permuted": mean_score,
        "std_permuted": spread,
        "min_permuted": float(min(scores)),
        "max_permuted": float(max(scores)),
        "n_permutations": float(len(scores)),
        "reference_level": float(reference),
        "tolerance": float(tolerance),
        "mean_fraction_reassigned": float(np.mean(reassigned)),
    }
    if chance is not None:
        measured["measured_chance"] = float(chance)
    if init_ref is not None:
        measured["init_reference"] = float(init_ref)
    if real_arm_score is not None:
        measured["real_arm_score"] = float(real_arm_score)
    if best_ckpt_scores:
        measured[BEST_CHECKPOINT_KEY] = float(np.mean(best_ckpt_scores))

    detail = (
        "reference is the {} ({:.4f}); a permuted-label run is judged against the "
        "level its own initialisation already reaches, never against an assumed "
        "chance (MISTAKES.md #33/#44).".format(reference_source, reference)
    )
    if best_ckpt_scores:
        detail += (
            " The best-by-val reading is reported under '{}' and is NOT the verdict: "
            "under a null it selects the epoch closest to initialisation "
            "(MISTAKES.md #34).".format(BEST_CHECKPOINT_KEY)
        )
    if weak:
        detail += (
            " NOTE the null spread {:.4f} exceeds the tolerance {:.4f}: this split has "
            "too little resolving power for the control to settle the question. Report "
            "that as the finding -- do not widen the tolerance until it passes "
            "(MISTAKES.md #23).".format(spread, tolerance)
        )

    return ControlResult(
        name=name,
        passed=passed,
        expectation=(
            "mean permuted metric over {} permutations within {:g} of the measured {} "
            "({:.4f}){}".format(
                len(scores),
                tolerance,
                reference_source,
                reference,
                "" if real_arm_score is None else ", and clearly below the real arm",
            )
        ),
        measured=measured,
        detail=detail,
        extra={
            "reference_source": reference_source,
            "per_permutation": [float(s) for s in scores],
            "weak_evidence": bool(weak),
            "perturbation_level": "group" if groups_list is not None else "item",
        },
    )


# --------------------------------------------------------------------------- #
# 2. Overfit one batch
# --------------------------------------------------------------------------- #


def select_signal_batch(
    items: Sequence[Any],
    n: int,
    signal_fn: Optional[Callable[[Any], bool]] = None,
    *,
    require_all: bool = True,
) -> List[Any]:
    """Pick ``n`` items that actually carry signal, for an overfit-one-batch control.

    Encodes MISTAKES.md #66 (subfolder103). The conventional ``train_items[:n]`` picked
    7 empty slices and 1 tumour slice on a 65-percent-empty segmentation corpus (a
    lexicographic filename sort put ``_1, _10, _11`` before ``_2``). The pooled Dice
    metric was hard-gated at exactly 0.0000 for the whole run even though the loss fell
    cleanly and the one foreground slice reached max probability 0.442 -- a control that
    looks like a wiring bug but is a batch-composition artefact.

    ``signal_fn`` is caller-supplied because "has signal" is problem-specific: more than
    200 foreground pixels, at least one box, a non-majority class label. When
    ``signal_fn`` is None this is just ``items[:n]``, which is the documented-unsafe
    default -- pass a filter on any dataset with sparse positives.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if signal_fn is None:
        return list(items[:n])
    picked = [it for it in items if signal_fn(it)][:n]
    if require_all and len(picked) < n:
        raise AssertionError(
            "only {} of {} requested items pass signal_fn; the overfit-one-batch "
            "control would be diluted (MISTAKES.md #66). Loosen signal_fn or lower n "
            "deliberately -- do not fall back to items[:n].".format(len(picked), n)
        )
    return picked


def overfit_one_batch_generic(
    fit_score: Callable[[], float],
    *,
    target: float,
    higher_is_better: bool = True,
    name: str = "overfit_one_batch",
) -> ControlResult:
    """Framework-free escape hatch: caller trains on one batch and returns one number.

    Use this for a non-torch arm (a GBM, an sklearn estimator, a classical pipeline
    with a fitted stage): fit the model on a handful of samples with regularisation
    disabled and return the TRAIN metric. The torch harness below is more useful when
    it applies, because it also records the loss trajectory that MISTAKES.md #49 and
    #52 both needed to diagnose their real cause.
    """
    score = float(fit_score())
    passed = score >= target if higher_is_better else score <= target
    return ControlResult(
        name=name,
        passed=passed,
        expectation="train metric on a single batch {} {}".format(
            ">=" if higher_is_better else "<=", target
        ),
        measured={"train_metric": score, "target": float(target)},
        detail=(
            "failure means the loss, the label mapping, the optimiser, or a frozen "
            "parameter is wired wrong -- fix that before reading any validation number"
        ),
    )


def overfit_one_batch(
    model_factory: Callable[[], Any],
    loss_fn: Callable[[Any, Any], Any],
    batch: Any,
    *,
    control_lr: float,
    control_steps: int,
    max_loss: Optional[float] = None,
    score_fn: Optional[Callable[[Any, Any], float]] = None,
    target: Optional[float] = None,
    device: str = "cpu",
    grad_clip_norm: Optional[float] = 1.0,
    optimizer_factory: Optional[Callable[[Any, float], Any]] = None,
    seed: int = 0,
    force_determinism: bool = True,
    record_every: int = 1,
    name: str = "overfit_one_batch",
) -> ControlResult:
    """Train a fresh model on ONE fixed batch and require it to memorise the batch.

    A model that cannot memorise a handful of samples has a structural bug -- a wrong
    label mapping, a detached graph, a frozen tower, a mis-shaped loss. This is the
    cheapest pre-training check there is (#4 subfolder70 added it after a BLIP text
    tower turned out to be randomly initialised).

    Three hard-won design points, all encoded here rather than left to the caller:

    * ``control_lr`` and ``control_steps`` are REQUIRED, explicit parameters
      (MISTAKES.md #52). A control's hyperparameters are tuned against one specific
      configuration, not a property of the control. subfolder95's 3e-3 control rate was
      calibrated for an 8-block network; re-run unchanged on a 16-block network it
      DIVERGED (final loss 0.0355 above the 0.02 gate, but min loss 0.0067 -- it had
      converged and then blown up). Re-calibrate on any capacity change.
    * ``grad_clip_norm`` defaults to 1.0 (MISTAKES.md #49). Without clipping,
      subfolder95's control fell cleanly to 0.022 by step 2000 then exploded to 22.4 at
      step 2200 -- a real training instability that the control's own run-to-run
      variance had been hiding. Both the real training loop and the control need it.
    * The body runs inside :func:`preserve_determinism_flags` (MISTAKES.md #51), so the
      determinism this control turns on for itself does not leak into the real training
      run an orchestrator calls next in the same process.

    Both ``final_loss`` and ``min_loss`` are reported, and the full trajectory is in
    ``extra["loss_trajectory"]``. When a control's number varies between nominally
    identical runs, force determinism and read the TRAJECTORY before touching the
    threshold: the variance is usually the symptom, not the noise (#49).

    Pass ``max_loss`` (loss below a threshold) and/or ``score_fn`` + ``target``
    (a metric on the same batch). At least one must be given.
    """
    if max_loss is None and (score_fn is None or target is None):
        raise ValueError("pass max_loss, or both score_fn and target -- there is no default gate")
    if control_steps <= 0:
        raise ValueError("control_steps must be positive")

    import torch  # lazy: this package must import on machines without torch

    with preserve_determinism_flags():
        if force_determinism:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        seed_everything(seed)

        model = model_factory().to(device)
        model.train()
        if optimizer_factory is None:
            opt = torch.optim.AdamW(model.parameters(), lr=float(control_lr))
        else:
            opt = optimizer_factory(model.parameters(), float(control_lr))

        inputs, targets = batch
        if hasattr(inputs, "to"):
            inputs = inputs.to(device)
        if hasattr(targets, "to"):
            targets = targets.to(device)

        trajectory: List[float] = []
        for step in range(int(control_steps)):
            opt.zero_grad(set_to_none=True)
            out = model(inputs)
            loss = loss_fn(out, targets)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            opt.step()
            if step % max(1, int(record_every)) == 0 or step == int(control_steps) - 1:
                trajectory.append(float(loss.detach().cpu().item()))

        final_loss = trajectory[-1] if trajectory else float("nan")
        min_loss = float(min(trajectory)) if trajectory else float("nan")

        score = None
        if score_fn is not None:
            model.eval()
            with torch.no_grad():
                score = float(score_fn(model, (inputs, targets)))

    checks: Dict[str, bool] = {}
    if max_loss is not None:
        checks["final_loss_below_max"] = bool(final_loss <= float(max_loss))
    if score is not None and target is not None:
        checks["metric_meets_target"] = bool(score >= float(target))
    passed = all(checks.values())

    measured: Dict[str, float] = {
        "final_loss": float(final_loss),
        "min_loss": float(min_loss),
        "control_lr": float(control_lr),
        "control_steps": float(control_steps),
        "steps_recorded": float(len(trajectory)),
    }
    if max_loss is not None:
        measured["max_loss"] = float(max_loss)
    if score is not None:
        measured["train_metric"] = float(score)
    if target is not None:
        measured["target"] = float(target)

    detail = (
        "failure means the loss, the label mapping, the optimiser, or a frozen "
        "parameter is wired wrong -- fix that before reading any validation number."
    )
    if not passed and min_loss < final_loss * 0.5:
        detail += (
            " NOTE min_loss {:.4g} is far below final_loss {:.4g}: the run CONVERGED "
            "and then diverged. That is a training instability (MISTAKES.md #49/#52), "
            "not a mis-set threshold -- lower control_lr for this capacity and check "
            "gradient clipping rather than loosening the gate.".format(min_loss, final_loss)
        )

    return ControlResult(
        name=name,
        passed=passed,
        expectation="; ".join(
            filter(
                None,
                [
                    None if max_loss is None else "final loss <= {:g}".format(max_loss),
                    None if target is None else "train metric >= {:g}".format(target),
                ],
            )
        ),
        measured=measured,
        detail=detail,
        extra={"checks": checks, "loss_trajectory": trajectory},
    )
