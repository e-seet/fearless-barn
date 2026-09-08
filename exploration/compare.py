#!/usr/bin/env python3
"""Rank the candidate arms on the VALIDATION metric and report the winner's TEST metric.

RMSE is lower-is-better, so the ranking here is a min, not the scaffold's max --
a detail worth stating because the generated template ranks by max and silently
inverting a metric's direction is exactly the kind of error that survives review.

Two gates fire here (root CLAUDE.md, Mode 2 step 4):
  * `declare_selection_set(..., selected_on="val")` -- raises if a future edit
    ever ranks on a test-labelled field.
  * `assert_training_pool_equality` -- every arm compared must have trained on
    the same rows, or the gap being reported is a data difference, not a method
    difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import dataset as D
from common.bootstrap import (assert_training_pool_equality, declare_selection_set,
                              run_dir, write_json_guarded)

ROOT = Path(__file__).resolve().parent
ARMS = list(config.ARMS)   # one list, in config (see config.ARMS)


def collect(split: str, featureset: str, seed: int, outlier_policy: str) -> "list[dict]":
    rows = []
    for arm in ARMS:
        # run_dir(), not string formatting -- one rename of the keying scheme
        # would otherwise leave this silently finding nothing and printing "skip".
        d = run_dir(config.ARTIFACTS / "runs", arm,
                    protocol=config.protocol_key(split, featureset, outlier_policy),
                    seed=seed, create=False)
        val_p, test_p = d / "eval_val.json", d / "eval_test.json"
        if not val_p.exists():
            print("  (skip {}: no eval_val.json at {})".format(arm, d))
            continue
        val = json.loads(val_p.read_text())
        model = json.loads((d / "model.json").read_text()) if (d / "model.json").exists() else {}
        row = {"arm": arm, "split": split, "featureset": featureset, "seed": seed,
               "outlier_policy": outlier_policy,
               "val_rmse": val["rmse"],
               # design-gate S2: the pooled RMSE is dominated by a few dozen rows
               # (worst 0.1% carry ~43% of SSE). The in-band figure is the
               # low-variance companion used to sanity-check the ranking.
               "val_rmse_in_band": val.get("rmse_in_band"),
               "val_lb_weighted_rmse": val.get("leaderboard_weighted_rmse"),
               "val_trivial_mean_rmse": val["trivial_baselines"]["train_mean"]["rmse"],
               "train_pool_ids_sha": model.get("train_pool_ids_sha"),
               # closing-gate finding: the run's feature-code version. S6 keyed
               # the feature CACHE on this hash but nothing keyed the RUNS, so a
               # comparison could rank arms built from different features.py
               # revisions and call the difference "method". n_features would
               # only catch a change that alters the COUNT.
               "feature_fingerprint": (model.get("feature_fingerprint")
                                       or val.get("feature_fingerprint")),
               "weights_are_val_fitted": model.get("weights_are_val_fitted"),
               # convergence_evidence() already writes this into every model.json;
               # until now nothing read it (MISTAKES.md #82). An arm that ran out
               # of epochs is not comparable with one that converged.
               "hit_max_epochs": (model.get("convergence") or {}).get("hit_max_epochs"),
               "any_airport_hit_round_cap": model.get("any_airport_hit_round_cap"),
               "n_train": model.get("n_train"),
               "result_path": str(val_p.relative_to(ROOT))}
        if test_p.exists():
            t = json.loads(test_p.read_text())
            row["test_rmse"] = t["rmse"]
            row["test_lb_weighted_rmse"] = t.get("leaderboard_weighted_rmse")
            row["test_trivial_mean_rmse"] = t["trivial_baselines"]["train_mean"]["rmse"]
        rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--skip-pool-check", action="store_true",
                    help="only for a partial run; the gate is on by default")
    ap.add_argument("--allow-unconverged", action="store_true",
                    help="record a deliberate exception for an arm that hit its "
                         "epoch/round cap; the gate is on by default")
    ap.add_argument("--skip-feature-check", action="store_true",
                    help="only for a deliberately mixed-code comparison; the "
                         "feature-version gate is on by default")
    ap.add_argument("--allow-seasonal-selection", action="store_true",
                    help="override the guard that refuses to crown a winner on "
                         "the anti-causal 'seasonal' protocol")
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES),
                    help="how the TRAINING tail is handled; part of the artifact "
                         "key. Evaluation sets are never filtered.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cands = collect(args.split, args.featureset, args.seed, args.outlier_policy)
    if not cands:
        print("no candidate results found for split={} featureset={} seed={}".format(
            args.split, args.featureset, args.seed))
        return 1

    # ---- gate 1: every arm trained on the same rows ----
    # This compared `ids` against ITSELF until the closing gate -- a list against
    # a copy of itself can never fail (MISTAKES.md #56, and design-gate B2, which
    # recorded each arm's own fingerprint but never got round to comparing them).
    # The real check is each arm's OWN recorded hash of the rows it consumed,
    # compared pairwise against a reference the comparer recomputes itself.
    pool_check = None
    if not args.skip_pool_check:
        _, ytr, meta = D.load_part(args.split, "train", args.featureset,
                                   filter_target=True,
                                   outlier_policy=args.outlier_policy)
        ref_sha = D.pool_fingerprint(meta.index)
        missing = [c["arm"] for c in cands if not c.get("train_pool_ids_sha")]
        if missing:
            raise SystemExit(
                "training-pool gate: arm(s) {} recorded no train_pool_ids_sha. An "
                "arm that cannot say which rows it consumed cannot be compared "
                "against one that can -- re-run those arms.".format(missing))
        disagree = {c["arm"]: c["train_pool_ids_sha"] for c in cands
                    if c["train_pool_ids_sha"] != ref_sha}
        if disagree:
            raise SystemExit(
                "training-pool gate FAILED: reference pool {} (n={}) but "
                "{} -- the gap between these arms is a DATA difference, not a "
                "method difference.".format(ref_sha, len(meta), disagree))
        pool_check = assert_training_pool_equality(
            sorted(meta.index.tolist()), sorted(meta.index.tolist()),
            name_a="reference_pool", name_b="reference_pool")
        pool_check["reference_sha"] = ref_sha
        pool_check["arm_shas"] = {c["arm"]: c["train_pool_ids_sha"] for c in cands}
        pool_check["method"] = ("each arm's own recorded hash of the rows it "
                                "consumed, vs a reference recomputed here")
        print("  [PASS] training-pool equality: {} arms all recorded pool {} "
              "(n={})".format(len(cands), ref_sha, len(meta)))

    # ---- gate 3: every arm actually converged ----
    # `max_epochs` is a safety cap, not a stopping condition: hitting it means the
    # model was still improving when the loop ended, so its number understates the
    # method (reproducibility-compute.md). MISTAKES.md #82 -- an epoch cap tuned on
    # one split's pool size silently truncated training on a 25% smaller pool, and
    # the only tell was this boolean, which nothing was reading.
    not_converged = [c["arm"] for c in cands
                     if c.get("hit_max_epochs") or c.get("any_airport_hit_round_cap")]
    if not_converged and not args.allow_unconverged:
        raise SystemExit(
            "convergence gate: arm(s) {} hit their epoch/round cap without early "
            "stopping firing. Their numbers understate the method and are not "
            "comparable with arms that converged -- raise the cap and re-run, or "
            "pass --allow-unconverged to record a deliberate exception."
            .format(not_converged))
    if not_converged:
        print("  [WARN] convergence gate overridden for {}".format(not_converged))

    # ---- gate 2: every arm was built from the same features.py ----
    fps = {c["arm"]: c.get("feature_fingerprint") for c in cands}
    unstamped = [a for a, f in fps.items() if not f]
    if unstamped and not args.skip_feature_check:
        raise SystemExit(
            "feature-version gate: arm(s) {} carry no feature_fingerprint, so this "
            "comparison cannot show they were built from the same features.py. "
            "Re-run them (the stamp is written by common/bootstrap.py). "
            "--skip-feature-check documents an intentional exception."
            .format(unstamped))
    distinct = sorted({f for f in fps.values() if f})
    if len(distinct) > 1 and not args.skip_feature_check:
        raise SystemExit(
            "feature-version gate FAILED: arms were built from DIFFERENT features.py "
            "revisions {} -- the gap being reported is partly a feature-code "
            "difference, not a method difference. Re-run every arm at one code "
            "version.".format(fps))
    feature_check = {"gate": "feature_code_agreement",
                     "fingerprints": fps,
                     "distinct": distinct,
                     "current_code": config.feature_code_fingerprint(),
                     "passed": bool(len(distinct) <= 1 and not unstamped)}
    if feature_check["passed"]:
        stale = (distinct and distinct[0] != feature_check["current_code"])
        print("  [{}] feature-code agreement: all arms at {}{}".format(
            "PASS" if not stale else "WARN", distinct[0] if distinct else "n/a",
            "" if not stale else " -- but features.py has changed since "
            "(now {}); the ranking is internally consistent yet stale".format(
                feature_check["current_code"])))
        feature_check["stale_vs_current_code"] = bool(stale)

    # ---- guard: the 'seasonal' protocol is anti-causal (design-gate S1) ----
    # Its test months are Jan + Jul 2025 while its training pool runs Feb-Nov, so
    # the Jan test rows PRECEDE the entire training set -- the model uses the
    # future to predict the past, and Jul is seasonally interpolated between two
    # training months. The real task extrapolates 13 months forward. It is a
    # composition probe, not a selection surface.
    selection_valid = args.split != "seasonal" or args.allow_seasonal_selection
    if not selection_valid:
        print("\n  NOTE: split 'seasonal' is anti-causal -- reporting numbers but "
              "NOT crowning a winner. Use it as a composition probe; select on "
              "'forward'. (--allow-seasonal-selection overrides.)")

    sel = declare_selection_set("compare.py winner", selected_on="val",
                                value=[c["arm"] for c in cands])
    winner = min(cands, key=lambda c: c["val_rmse"])   # RMSE: lower is better

    # ---- caveat: arms that FIT free parameters on the val labels ----
    # config.VAL_FITTED_ARMS documents which and why. Their val RMSE is in-sample
    # for those parameters, so a crown they win over a non-val-fitted arm is not
    # established by this ranking alone.
    # Per-RUN truth, not just the static arm list: `cross_blend` fits its weights
    # on val by default but on an inner holdout with --weight-split, and only the
    # first is val-fitted. An arm-level list cannot express that, and stamping a
    # clean run with the caveat is as wrong as omitting it from a dirty one.
    val_fitted = []
    for c in cands:
        flag = c.get("weights_are_val_fitted")
        if flag is True or (flag is None and c["arm"] in config.VAL_FITTED_ARMS):
            val_fitted.append(c["arm"])
    honest = [c for c in cands if c["arm"] not in val_fitted]
    winner_not_val_fitted = min(honest, key=lambda c: c["val_rmse"]) if honest else None
    if val_fitted:
        print("\n  NOTE: {} fit free parameters directly on the val labels "
              "(config.VAL_FITTED_ARMS), so their val score is in-sample for "
              "those parameters.".format(", ".join(val_fitted)))
        if winner_not_val_fitted is not None:
            print("        best arm that did NOT fit on val: {} "
                  "(val {:.2f})".format(winner_not_val_fitted["arm"],
                                        winner_not_val_fitted["val_rmse"]))

    print("\nranked on val RMSE (lower is better; never ranked on test):")
    for c in sorted(cands, key=lambda c: c["val_rmse"]):
        tst = "{:8.2f}".format(c["test_rmse"]) if "test_rmse" in c else "     n/a"
        print("  {:16s} val={:8.2f}  test={}  (val trivial-mean floor {:.2f})".format(
            c["arm"], c["val_rmse"], tst, c["val_trivial_mean_rmse"]))
    print("\nwinner (val-selected): {}".format(winner["arm"]))
    if "test_rmse" in winner:
        print("  test RMSE            : {:.2f} s".format(winner["test_rmse"]))
        print("  test trivial floor   : {:.2f} s".format(winner["test_trivial_mean_rmse"]))
        if winner.get("test_lb_weighted_rmse"):
            print("  test LB-weighted RMSE: {:.2f} s".format(winner["test_lb_weighted_rmse"]))
    else:
        print("  test RMSE            : NOT YET COMPUTED -- not registerable")

    # design-gate S9: leaderboard_weighted_rmse corrects the AIRPORT mix only.
    # The scored set is also 71% Jan / 29% Jul, and EDDF/EGLL/EHAM are
    # over-weighted precisely because they appear in both months -- so on a
    # Nov-Dec test set that reweighting attributes a month-composition effect to
    # airports. Recorded so the number is never read as a leaderboard prediction.
    out = {"selection": dict(sel), "selection_valid": bool(selection_valid),
           "selection_invalid_reason": (
               None if selection_valid else
               "split 'seasonal' is anti-causal: Jan test rows precede the whole "
               "training pool and Jul is seasonally interpolated between training "
               "months; it is a composition probe, not a selection surface"),
           "lb_weighting_caveat": (
               "leaderboard_weighted_rmse reweights by AIRPORT only, not by "
               "airport|month. It is an approximation on any part whose month mix "
               "differs from the scored 71% Jan / 29% Jul."),
           "pool_check": pool_check, "feature_check": feature_check,
           "convergence_check": {"gate": "all_arms_converged",
                                 "not_converged": not_converged,
                                 "overridden": bool(args.allow_unconverged),
                                 "passed": bool(not not_converged)},
           "val_fitted_arms": val_fitted,
           "val_fitted_caveat": (
               None if not val_fitted else
               "arm(s) {} fit free parameters (blend weights) directly on the val "
               "labels, so their val RMSE is in-sample for those parameters and is "
               "not comparable at face value against arms that only early-stopped "
               "on val. See config.VAL_FITTED_ARMS.".format(val_fitted)),
           "winner_not_val_fitted": (
               None if winner_not_val_fitted is None else {
                   "arm": winner_not_val_fitted["arm"],
                   "val_rmse": winner_not_val_fitted["val_rmse"],
                   "test_rmse": winner_not_val_fitted.get("test_rmse")}),
           "candidates": cands,
           "winner": winner["arm"], "winner_val_rmse": winner["val_rmse"],
           "winner_test_rmse": winner.get("test_rmse")}
    write_json_guarded(
        ROOT / "compare_summary__{}__seed{}.json".format(
            config.protocol_key(args.split, args.featureset, args.outlier_policy),
            args.seed),
        out, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
