#!/usr/bin/env python3
"""Arm cross_blend: convex blend of the per-airport family and the neural family.

Why — measured, not assumed
---------------------------
`stage8_triage.py`'s band decomposition showed the two families fail in
*complementary* places on forward/test:

| band | rows | per_airport_stack SSE | ensemble SSE | better |
|---|---|---|---|---|
| in-band (y<=7200) | 327,907 | **1.638e10** | 2.075e10 | per-airport, by 4.4e9 |
| mid-tail | 58 | 1.155e9 | **6.13e8** | neural |
| artifact (y>=80000) | 3 | 1.023e10 | **2.60e9** | neural |

Routing each band to its best arm would score 240.82 against the best single
arm's 270.36 — a 29.5 s oracle gap. That upper bound is not reachable (it picks
per band using the labels), but it establishes the complementarity is large and
real rather than a rounding effect.

A plain convex sweep confirms an interior optimum on **both** surfaces: val
254.40 -> 253.08 at w=0.2, test 270.36 -> 263.21 at w=0.4, where w is the weight
on `per_airport_stack`. A weight fitted near the val optimum lands at test
~264.5, about 5.9 s better than either parent.

Two weight modes
----------------
* ``global`` — one scalar. The minimum-optimism version, and exactly what the
  probe above measured.
* ``per_airport`` — one scalar per airport. This is band-aware routing expressed
  through the axis that happens to carry the bands: the tail lives almost
  entirely in LIRF, so LIRF should lean neural while the other nine lean
  per-airport. Ten parameters instead of one.

Two ways the weights can be fitted — and only one of them is clean
------------------------------------------------------------------
**Default (no ``--weight-split``): fitted on the validation labels.** That is the
same thing this build flags `ensemble_antigravity` for, so this arm is in
`config.VAL_FITTED_ARMS` too and `compare.py` stamps the same caveat. It is not
exempt because its author noticed the problem. The magnitude argument — offered
as a reason the caveat is *small* here, not a reason to skip it — is that ten
least-squares weights on 369,624 validation rows carry optimism of order
sigma^2 * k / n, negligible against a ~250 s RMSE. That is a difference in degree
from a 5-weight NNLS blend, not a difference in kind. **The val number this mode
reports is in-sample for its weights.**

**``--weight-split forward_inner``: the clean version.** Weights are solved on
Jul-Aug 2025 using arms trained on Jan-Jun only (`config.SPLITS["forward_inner"]`),
then applied to the `forward` models. No model whose predictions are being
weighted has seen those rows, and the weights never touch the validation labels
this arm is ranked on — so the reported val RMSE is genuinely held out and the arm
is comparable with every other arm at face value. `model.json` records
``weights_are_val_fitted`` so a reader never has to infer which mode produced a
number.

The cost of the clean version is that the weights come from models trained on
6 months rather than 8. `per_airport_stack` already showed what that bias does:
its one-month inner holdout under-weighted the segment models because they suffer
more from less data, and doubling the holdout fixed it. The same bias applies
here and is the thing to watch when comparing the two modes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)
from pipeline_per_airport.model import AIRPORT_COL

ARM = "cross_blend"
DEFAULT_A = "per_airport_stack"
DEFAULT_A_TAG = "inner2"
DEFAULT_B = "ensemble_antigravity"


def solve_w(y, pa, pb) -> float:
    """Least-squares weight on arm A, clipped to [0, 1] (see per_airport_stack)."""
    d = np.asarray(pa, dtype="float64") - np.asarray(pb, dtype="float64")
    r = np.asarray(y, dtype="float64") - np.asarray(pb, dtype="float64")
    den = float(np.dot(d, d))
    if den <= 0 or not np.isfinite(den):
        return 0.0
    return float(np.clip(np.dot(d, r) / den, 0.0, 1.0))


def apply_weights(pa, pb, weights, ap=None) -> np.ndarray:
    if isinstance(weights, float):
        return weights * pa + (1.0 - weights) * pb
    out = np.empty(len(pa), dtype="float64")
    for a in pd.unique(ap):
        m = ap == a
        w = float(weights.get(str(a), 0.0))
        out[m] = w * pa[m] + (1.0 - w) * pb[m]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--arm-a", default=DEFAULT_A, choices=list(config.ARMS))
    ap.add_argument("--arm-a-tag", default=DEFAULT_A_TAG)
    ap.add_argument("--arm-b", default=DEFAULT_B, choices=list(config.ARMS))
    ap.add_argument("--arm-b-tag", default=None)
    ap.add_argument("--weight-mode", default="per_airport",
                    choices=["global", "per_airport"])
    ap.add_argument("--weight-split", default=None,
                    help="fit the weights on ANOTHER split's val using arms "
                         "trained on that split (e.g. forward_inner), then apply "
                         "them to this split's models. This is the clean version: "
                         "the weights never see the validation labels this arm is "
                         "ranked on, so the reported val RMSE is genuinely "
                         "held-out and the arm is comparable at face value. "
                         "Without it the weights are fitted on val and the arm is "
                         "declared in config.VAL_FITTED_ARMS.")
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--freeze-from", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    refit_only = not config.SPLITS[args.split]["val"]
    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)

    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set. Pass --freeze-from.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit("--freeze-from points at a non-selection surface.")
        weights = (float(frozen["weights"]) if frozen["weight_mode"] == "global"
                   else {str(k): float(v) for k, v in frozen["weights"].items()})
        val_rmse = None
        print("[{}] weights frozen from {}".format(ARM, args.freeze_from))
        sweep = frozen.get("sweep")
    else:
        # --- the surface the WEIGHTS are solved on ---
        wsplit = args.weight_split or args.split
        if args.weight_split:
            if args.weight_split == args.split:
                raise SystemExit(
                    "--weight-split must name a DIFFERENT split; fitting on this "
                    "split's own val is the val-fitted mode (omit the flag).")
            Xw_tr, _, _ = D.load_part(wsplit, "train", args.featureset,
                                      filter_target=True,
                                      outlier_policy=args.outlier_policy)
            Xw, yw, meta_w = D.load_part(wsplit, "val", args.featureset,
                                         filter_target=False)
            yw = np.asarray(yw, dtype="float64")
            pa_w = predict_arm(args.arm_a, wsplit, "val", args.featureset, args.seed,
                               args.outlier_policy, X=Xw, meta=meta_w, Xtr=Xw_tr,
                               tag=args.arm_a_tag)
            pb_w = predict_arm(args.arm_b, wsplit, "val", args.featureset, args.seed,
                               args.outlier_policy, X=Xw, meta=meta_w, Xtr=Xw_tr,
                               tag=args.arm_b_tag)
            print("[{}] WEIGHTS fitted on '{}' val (n={}), models trained on '{}': "
                  "{} = {:.3f} | {} = {:.3f}".format(
                      ARM, wsplit, len(yw), wsplit, args.arm_a, rmse(yw, pa_w),
                      args.arm_b, rmse(yw, pb_w)))

        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset,
                                        filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pa = predict_arm(args.arm_a, args.split, "val", args.featureset, args.seed,
                         args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr,
                         tag=args.arm_a_tag)
        pb = predict_arm(args.arm_b, args.split, "val", args.featureset, args.seed,
                         args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr,
                         tag=args.arm_b_tag)
        if not args.weight_split:
            yw, pa_w, pb_w, meta_w = yva, pa, pb, meta_va
        print("[{}] base arms on {} val: {} = {:.3f} | {} = {:.3f}".format(
            ARM, args.split, args.arm_a, rmse(yva, pa), args.arm_b, rmse(yva, pb)))

        ap_v = meta_va[AIRPORT_COL].astype("object").to_numpy()
        ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()
        if args.weight_mode == "global":
            weights = solve_w(yw, pa_w, pb_w)
            print("   global w({}) = {:.4f}".format(args.arm_a, weights))
        else:
            weights = {}
            for a in sorted(pd.unique(ap_w)):
                mw = ap_w == a
                weights[str(a)] = solve_w(yw[mw], pa_w[mw], pb_w[mw])
                m = ap_v == a
                if not m.any():
                    continue
                blend_a = apply_weights(pa[m], pb[m], weights[str(a)])
                print("   {:6s} n_w={:6d} w={:.3f}   {} {:8.2f} | {} {:8.2f} | blend {:8.2f}"
                      .format(str(a), int(mw.sum()), weights[str(a)],
                              args.arm_a[:12], rmse(yva[m], pa[m]),
                              args.arm_b[:12], rmse(yva[m], pb[m]),
                              rmse(yva[m], blend_a)))
            missing = sorted(set(map(str, pd.unique(ap_v))) - set(weights))
            if missing:
                raise SystemExit(
                    "airport(s) {} appear in {}/val but had no weight solved on "
                    "{}/val -- routing would fall back to 0.0 (pure arm B) "
                    "silently.".format(missing, args.split, wsplit))
        blend = np.clip(apply_weights(pa, pb, weights, ap_v),
                        config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse = rmse(yva, blend)
        print("[{}] val RMSE (clipped) = {:.3f}   {}".format(
            ARM, val_rmse,
            "<-- HELD OUT (weights fitted on '{}')".format(wsplit) if args.weight_split
            else "<-- IN-SAMPLE for the weights"))
        sweep = [{"w": float(w), "val_rmse": rmse(yva, np.clip(
            w * pa + (1 - w) * pb, config.PREDICTION_CLIP_MIN_SEC,
            config.PREDICTION_CLIP_MAX_SEC))} for w in np.arange(0, 1.01, 0.1)]

    sel = declare_selection_set(
        "cross_blend weights ({})".format(args.weight_mode),
        selected_on="frozen" if refit_only else "val", value=weights)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed,
        "base_arms": {"a": args.arm_a, "a_tag": args.arm_a_tag,
                      "b": args.arm_b, "b_tag": args.arm_b_tag},
        "weight_mode": args.weight_mode, "weights": weights,
        "weight_split": args.weight_split,
        "weights_are_val_fitted": bool(not args.weight_split),
        "weight_note": ("w is the weight on arm A. Fitted on the VALIDATION "
                        "labels: this arm is in config.VAL_FITTED_ARMS and its "
                        "val RMSE is IN-SAMPLE for these weights, not a held-out "
                        "estimate. See the module docstring."),
        "n_weights": (1 if args.weight_mode == "global" else len(weights)),
        "val_rmse_clipped": val_rmse, "w_sweep": sweep,
        "refit_only": bool(refit_only), "selection": dict(sel),
        "n_train": int(len(Xtr)),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": args.outlier_policy,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
