"""The three-question result triage, as one function with a three-valued verdict.

Canonical implementation of methodology/common/production-readiness.md ->
"Triage the result before you believe it". Encodes MISTAKES.md entries #26, #32, #45,
#47, #68.

The three questions are CONJUNCTIVE and measure different things:

  q1  Is the number good enough?      (bar checks against the plan's declared floors)
  q2  Is the number real?             (believability triggers + a leak instrument)
  q3  Improve or close?               (three-valued, and q2 can veto q1)

MISTAKES.md #32 is the reason q3 is not ``all(q1_checks)``. subfolder88 implemented
the three questions literally, then computed q3 as ``all(q1_checks.values())``,
ignoring q2 entirely. On the first real read the winner scored test mAP@10 0.9905
against val 0.9897 -- test ABOVE val and above the near-perfect threshold -- so q2
correctly set ``leak_instrument_triggered: true``. q3 nevertheless printed
"CLOSE -- bar cleared", which is exactly the string that gets pasted into a
RETROSPECTIVE and an INDEX row. A result can pass every threshold in the plan
BECAUSE it is contaminated; that is the whole reason q2 exists as a separate step.

The trigger is discharged ONLY by a passing leak instrument (a label-permutation
control retrained on shuffled labels, a re-run duplicate audit on the split as built)
-- never by q1 passing, and never by the trigger being judged implausible.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = ["triage", "CLOSE", "HOLD", "BELOW_BAR"]

CLOSE = "CLOSE"
HOLD = "HOLD"
BELOW_BAR = "BELOW-BAR"


def triage(
    *,
    metric_name: str,
    test_metric: float,
    val_metric: Optional[float] = None,
    trivial_baseline: Optional[float] = None,
    trivial_baseline_name: str = "trivial baseline",
    bar_checks: Mapping[str, bool],
    expected_band: Optional[Sequence[float]] = None,
    leak_instrument_passed: Optional[bool] = None,
    leak_instrument_name: str = "label-permutation control",
    ceiling: float = 0.99,
    val_minus_test_tolerance: float = 0.0,
    extra_triggers: Iterable[str] = (),
    synthetic: bool = False,
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Run all three triage questions and return a verdict dict.

    Parameters
    ----------
    bar_checks:
        q1. ``{check_name: passed}``, one entry per criterion the PLAN named -- the
        declared metric floor, per-class recall floors, a latency budget. Compute the
        booleans at the call site so the numbers stay visible in the caller's artifact.
    expected_band:
        q2's strongest instrument: the ``(low, high)`` band the plan pre-registered for
        this metric. Leave it None only if you genuinely have none -- an unset band
        emits an explicit "check unavailable" TRIGGER rather than silently passing
        (MISTAKES.md #26: subfolder86's band was deliberately set to null because the
        only distribution available at design time was the INCUMBENT SYSTEM's own
        output, and a plausibility band calibrated against the system you are replacing
        is a leak-detection instrument borrowing its notion of "plausible" from the
        thing under test).
    leak_instrument_passed:
        None = the instrument has not been run. True = it ran and passed. False = it
        ran and failed. A fired trigger is discharged only by True.
    trivial_baseline:
        Report it next to the headline number, always. For a metric with no intuitive
        floor -- MOTA, a custom composite -- DERIVE the floor and state it (MISTAKES.md
        #47: MOTA with zero detections emitted is exactly 0, so subfolder94's -0.379
        meant "actively worse than emitting nothing", not merely "underperforms"). See
        ``baselines.stated_floor``.

    Returns a dict with ``q1_verdict`` / ``q2_believability`` / ``q3_improve_or_close``.
    ``q3_improve_or_close`` is one of CLOSE / HOLD / BELOW-BAR.
    """
    checks = {str(k): bool(v) for k, v in dict(bar_checks).items()}
    if not checks:
        raise ValueError(
            "bar_checks is empty -- q1 has nothing to test. Name the criteria the plan "
            "declared; a triage with no bar is not a triage."
        )
    all_q1 = all(checks.values())
    failed = [k for k, v in checks.items() if not v]

    triggers: List[str] = []

    band: Optional[Tuple[float, float]] = None
    if expected_band is None:
        triggers.append(
            "expected_band is UNSET, so the 'is this too good?' check is UNAVAILABLE -- "
            "this is a trigger, not a pass (MISTAKES.md #26). Derive the band from the "
            "measured label distribution plus a quick fit on train BEFORE the first "
            "real run, and never from another deployed system's output distribution."
        )
    else:
        lo, hi = float(expected_band[0]), float(expected_band[1])
        band = (lo, hi)
        if higher_is_better and test_metric > hi:
            triggers.append(
                "{} {:.4f} is ABOVE the pre-registered band [{:.4f}, {:.4f}] -- a result "
                "better than the plan expected is a leakage signal first and an "
                "achievement second".format(metric_name, test_metric, lo, hi)
            )
        if not higher_is_better and test_metric < lo:
            triggers.append(
                "{} {:.4f} is BELOW the pre-registered band [{:.4f}, {:.4f}] -- better "
                "than the plan expected; treat as a leakage signal".format(
                    metric_name, test_metric, lo, hi
                )
            )

    if higher_is_better and test_metric >= ceiling:
        triggers.append(
            "{} {:.4f} is at or above the near-perfect ceiling {:.4f}".format(
                metric_name, test_metric, ceiling
            )
        )

    if val_metric is not None:
        gap = float(val_metric) - float(test_metric)
        if higher_is_better and gap < -abs(val_minus_test_tolerance):
            triggers.append(
                "test ({:.4f}) is ABOVE val ({:.4f}) -- the held-out set scoring better "
                "than the set used for selection is the classic contamination "
                "signature".format(test_metric, val_metric)
            )
        if not higher_is_better and gap > abs(val_minus_test_tolerance):
            triggers.append(
                "test ({:.4f}) is BETTER than val ({:.4f}) on a lower-is-better "
                "metric".format(test_metric, val_metric)
            )

    if trivial_baseline is not None:
        b = float(trivial_baseline)
        if higher_is_better and b >= float(test_metric) - 1e-9:
            triggers.append(
                "the {} scores {:.4f} against the model's {:.4f} -- the headline metric "
                "has no dynamic range on this subset, so the comparison is a tautology "
                "rather than a verdict".format(trivial_baseline_name, b, test_metric)
            )

    if synthetic:
        triggers.append("SYNTHETIC data -- this number is a wiring check, not a result")

    for t in extra_triggers:
        triggers.append(str(t))

    triggered = bool(triggers)
    instrument_run = leak_instrument_passed is not None
    discharged = bool((not triggered) or (leak_instrument_passed is True))

    if all_q1 and discharged and not synthetic:
        verdict = CLOSE
        next_action = (
            "clears the bar and believability is discharged; run the repeat-seed sweep "
            "and the advisor closing gate before registering"
        )
    elif all_q1:
        verdict = HOLD
        next_action = (
            "bar cleared but the believability trigger fired and its leak instrument "
            "has {} -- run the {} and re-triage BEFORE this number is written "
            "anywhere".format(
                "not been run" if not instrument_run else "not passed", leak_instrument_name
            )
        )
    else:
        verdict = BELOW_BAR
        next_action = "below bar on {} -- name the specific lever and iterate".format(failed)

    return {
        "q1_verdict": {
            "metric": metric_name,
            "test": float(test_metric),
            "val": None if val_metric is None else float(val_metric),
            "trivial_baseline": None if trivial_baseline is None else float(trivial_baseline),
            "checks": checks,
            "all_checks_pass": all_q1,
            "failed_criteria": failed,
        },
        "q2_believability": {
            "expected_band": None if band is None else [band[0], band[1]],
            "expected_band_available": band is not None,
            "triggers": triggers,
            "leak_instrument_triggered": triggered,
            "leak_instrument": leak_instrument_name,
            "leak_instrument_run": instrument_run,
            "leak_instrument_passed": leak_instrument_passed,
            "discharged": discharged,
            "note": (
                "the trigger is discharged ONLY by a passing leak instrument, never by "
                "the q1 checks passing -- q1 asks whether the number is good enough, q2 "
                "asks whether it is real (MISTAKES.md #32)"
            ),
        },
        "q3_improve_or_close": verdict,
        "next_action": next_action,
        "verdict_line": (
            "{} test {:.4f}{}{} -- {}".format(
                metric_name,
                float(test_metric),
                "" if val_metric is None else " (val {:.4f})".format(float(val_metric)),
                "" if trivial_baseline is None
                else " vs {} {:.4f}".format(trivial_baseline_name, float(trivial_baseline)),
                verdict,
            )
        ),
    }
