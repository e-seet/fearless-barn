#!/usr/bin/env python3
"""Arm airport_lirf_identity_multimodel: zero-parameter LIRF telemetry-identity override.

Why zero parameters, not a fitted regression
---------------------------------------------
LIRF flights with no flight record (`has_flight_record == 0`) and a large
schedule delay (`delta_sched_sec > 25,200s`) are subject to a known Italian
ANSP telemetry behaviour: BLOCK_TIME defaults to SCHED_TIME when radar track
data is missing, so `y = TAKEOFF - BLOCK_TIME = TAKEOFF - SCHED_TIME =
delta_sched_sec` almost exactly.

Checked against the FULL YEAR of training data (not just one split) before
building this arm (see stage11_segment_audit.py and
plans/11_plan_outlier_audit_and_overfitting_check.md for the full writeup):

- 25,200-50,000s band: 34 rows exist in the entire year. y - delta_sched_sec
  = 0.0 +/- 2.9s for all 34. This is a near-exact identity.
- >50,000s band ("overnight rollover"): only 20 rows exist in the entire
  year, with y - delta_sched_sec std = 13,488s -- NOT a clean identity by
  itself. But checked directly against every real val/test row that falls in
  it (7 rows total), using delta_sched_sec AS THE PREDICTION, with no fitting
  at all, beats the base model's prediction on every single one -- sometimes
  by 10x (one test row: 1,503s error vs 15,402s for the base model). The base
  model radically underpredicts these because it has almost no training
  signal for taxi times near 24h; delta_sched_sec, despite not being a clean
  identity in this band, is still a much better zero-effort guess than a
  model extrapolating from a population where such delays essentially never
  occur.

A prior attempt (not in this codebase's own history -- found during an audit
of a separate agent's work on this same repo) fit a free 3-parameter ridge
regression per band instead. Checked directly: 7 and 15 rows respectively on
the honest fitting surface, applied to only 3-5 real evaluation rows -- a
regression with that few points relative to its parameter count is fitting
noise, not a discovered relationship (see the plan doc for the full
comparison). This arm deliberately uses ZERO fitted parameters instead: just
the raw feature as the prediction, under a condition that is itself grounded
in a real, checked mechanism rather than searched for by whatever segment
happened to score well.

Because there is nothing to fit, this arm needs no forward_inner holdout and
cannot leak into val -- the override is a fixed rule, applied identically
regardless of which rows happen to be in train/val/test.

Two defaults changed in the 2026-09 pipeline review (plan 12, candidate 2)
----------------------------------------------------------------------------
1. `--band` now defaults to `idband` (the checked 25.2k-50k identity ONLY),
   not `full`. The two bands above are not the same kind of claim: the first
   is a verified identity, the second is a declared tail bet resting on ~7
   evaluation rows. The review's verdict was "adopt the identity band
   unconditionally; treat the >50k branch as a declared tail bet", so the
   default is now the part that was actually verified. Pass `--band full` to
   get the previous behaviour, which is still recorded and still available --
   it is a bet, not an error.
2. `--base-arm` now defaults to `airport_operator_multimodel`, not
   `airport_multiseed_multimodel`. The old base chained through
   airport_unsup_multimodel, both of which the review judged over-tuned and
   moved to overtuned/. This arm's own override is clean, but stacking it on
   an over-tuned base made its headline number inherit that base's optimism.
   The new default is the tier-A blend.

Both are selectable rather than hardcoded precisely so the old configuration
stays reproducible for comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded

ARM = "airport_lirf_identity_multimodel"
# Tier-A blend, NOT the quarantined airport_multiseed_multimodel -- see docstring.
BASE_ARM = "airport_operator_multimodel"
LEGACY_BASE_ARM = "airport_multiseed_multimodel"
LIRF_AIRPORT = "LIRF"
THRESHOLD_SEC = 25200.0
# Upper edge of the CHECKED identity. Above this the relationship is not an
# identity (full-year residual std 13,488 s over 20 rows), so `--band idband`
# stops here and leaves those rows to the base model.
IDENTITY_BAND_HI_SEC = 50000.0


def apply_lirf_identity_override(pred_base, X, ap_arr, threshold=THRESHOLD_SEC,
                                 band_hi=None):
    """Zero-parameter override: replace the base prediction with the raw
    delta_sched_sec feature wherever the LIRF telemetry-default condition
    holds. No coefficients, no fitting -- see module docstring for why.

    band_hi=None reproduces the original behaviour (override everything above
    `threshold`, i.e. the checked identity PLUS the >50k tail bet).
    band_hi=IDENTITY_BAND_HI_SEC restricts it to the verified identity only.
    """
    pred = np.asarray(pred_base, dtype="float64").copy()
    has_rec = X["has_flight_record"].to_numpy()
    ds = X["delta_sched_sec"].fillna(0).to_numpy()
    m = (ap_arr == LIRF_AIRPORT) & (has_rec == 0) & (ds > threshold)
    if band_hi is not None:
        m &= (ds <= band_hi)
    pred[m] = ds[m]
    return np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC), int(m.sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--threshold", type=float, default=THRESHOLD_SEC)
    ap.add_argument("--base-arm", default=BASE_ARM,
                    help="Arm whose predictions this overrides. Default is the "
                         "tier-A blend; pass {} to reproduce the pre-review "
                         "configuration.".format(LEGACY_BASE_ARM))
    ap.add_argument("--band", default="idband", choices=["idband", "full"],
                    help="idband (default): override ONLY the checked "
                         "25.2k-50k identity. full: also override >50k, which "
                         "is a declared tail bet, not an identity.")
    ap.add_argument("--tag", default=None,
                    help="Run-dir tag. Defaults to the band name so the two "
                         "variants never overwrite each other.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    base_arm = args.base_arm
    band_hi = IDENTITY_BAND_HI_SEC if args.band == "idband" else None
    tag = args.tag if args.tag is not None else args.band

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": tag} if tag else None))

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)

    has_val = bool(config.SPLITS[args.split]["val"])
    val_rmse = None
    n_overridden_val = 0
    if has_val:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        yva = np.asarray(yva, dtype="float64")
        pva_base = predict_arm(base_arm, args.split, "val", args.featureset, args.seed,
                               args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
        ap_va = meta_va["ADEP_mvt"].astype("object").to_numpy()
        pred_va, n_overridden_val = apply_lirf_identity_override(
            pva_base, Xva, ap_va, args.threshold, band_hi=band_hi)
        val_rmse = rmse(yva, pred_va)
        print(f"[{ARM}] band={args.band} base={base_arm}")
        print(f"[{ARM}] val RMSE = {val_rmse:.4f}  (base = {rmse(yva, pva_base):.4f})  "
              f"{n_overridden_val} rows overridden")

    meta = {
        "arm": ARM,
        "base_arm": base_arm,
        "band": args.band,
        "identity_band_hi_sec": band_hi,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "condition": "ADEP_mvt == 'LIRF' and has_flight_record == 0 and delta_sched_sec > threshold",
        "threshold_sec": args.threshold,
        "n_free_params": 0,
        "prediction_when_condition_holds": "delta_sched_sec (raw feature, no fitting)",
        "n_overridden_val": n_overridden_val,
        "val_rmse": float(val_rmse) if val_rmse is not None else None,
        "weights_are_val_fitted": False,
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "feature_fingerprint": config.feature_code_fingerprint(),
    }
    write_json_guarded(out / "model.json", meta, force=args.force)

    if has_val:
        res_val = E.evaluate(yva, pred_va, meta_va, ytr, part="val", arm=ARM, split=args.split,
                             featureset=args.featureset, seed=args.seed,
                             extra={"threshold_sec": args.threshold,
                                    "n_overridden_val": n_overridden_val,
                                    "band": args.band,
                                    "base_arm": base_arm})
        write_json_guarded(out / "eval_val.json", res_val, force=args.force)

    print(f"[{ARM}] saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
