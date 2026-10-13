#!/usr/bin/env python3
"""Arm per_airport_stack: per-airport blend of the pooled and per-airport models.

Where this comes from -- CORRECTED, after checking the sources
-------------------------------------------------------------
This arm was originally justified as a transfer from the Mode 0 library, on the
claim that Santander and Facial Keypoints "combine per-group models with a global
model rather than replacing it". **That claim was wrong.** Re-reading the two
write-ups:

* `santander-.../sol2_per_feature_gbm_cnn_meta/WRITEUP.md` (lines 1, 53-56):
  200 per-**feature** LightGBMs plus a 1-D CNN meta-model. Each base model sees
  ONE feature and cannot produce a usable prediction alone, so recombination is
  structurally mandatory -- and the meta-model combines the 200 specialists with
  **each other**, not with any global model. Its partition axis is columns, not
  rows, which makes it a different structure from this one entirely.
* `facial-keypoints-detection/sol1_four_stage_pipeline/WRITEUP.md` (lines 26, 45):
  "ensemble: 7 models for the 4-kp subset, 5 for the 15-kp subset", and
  "per-keypoint-group specialist ensembles ... beat one all-keypoints [model]".
  That is an ensemble **within** each group, explicitly **replacing** the global
  model -- not a blend with one.

So every ROW-partitioned precedent in the library (M5 per-store, Facial Keypoints
per-schema, DAGM per-variant) *replaces* the global model. None blends with one.
The blend below is this build's own idea, and the correctly-read precedent
predicts it will not beat pure per-airport -- which is what happened
(val 283.85 vs 282.66).

It is kept as a candidate anyway because its result is informative rather than
merely negative (see the LIRF note below), and because a candidate that is
measured and loses is worth more than one that is assumed away.

**The lever that IS transferable, and is not this one:** Facial Keypoints
ensembles several models *inside* each group. The per-airport analogue is a
multi-seed / multi-config ensemble within each airport, not a blend with the
pooled model.

The measured reason to do it here: replacing the pooled model with per-airport
models improved eight airports on val and made **two worse** -- LEBL +1.58 s and
LFPG +1.68 s. A blend keeps the pooled model where it is the better of the two
instead of paying that cost for the segmentation everywhere.

Why the weights are NOT fitted on val
-------------------------------------
This build already carries one arm whose free parameters are fitted directly on
the validation labels (`config.VAL_FITTED_ARMS`), and its val score is in-sample
for those parameters as a result. Repeating that here would make this arm's val
number incomparable with every other arm's, on the one surface the winner is
chosen from.

So the blend weights come from an **inner, time-based holdout carved out of the
training pool**: the last training month is held back, both models are fitted on
the months before it, and one weight per airport is solved on that month. The
weights are then applied to the models trained on the full pool. Validation is
never touched, so this arm's val RMSE is a genuine held-out estimate and
`compare.py` can rank it against the others at face value.

The inner split is time-based, not random, for the same reason the outer one is:
the real task extrapolates forward, and a randomly-carved inner fold would let a
weight be tuned on the future.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import features as F
from common.metrics import rmse
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)
from pipeline_per_airport.model import PerAirportLGBM, AIRPORT_COL
from pipeline_lightgbm.train import PARAMS as POOLED_PARAMS
from pipeline_per_airport_tuned.train import GRID as LGB_GRID

ARM = "per_airport_stack"


def solve_alpha(y, p_seg, p_pool) -> float:
    """Least-squares weight on the per-airport model, clipped to [0, 1].

    minimise  sum (y - (a*p_seg + (1-a)*p_pool))^2
            = sum ((y - p_pool) - a*(p_seg - p_pool))^2
    => a* = <d, r> / <d, d>   with d = p_seg - p_pool, r = y - p_pool.

    Clipped because an unclipped weight outside [0, 1] is extrapolation beyond
    both models, which on a handful of rows is fitting noise rather than
    combining evidence. A degenerate cell (the two models agree exactly, so
    <d,d> == 0) falls back to the pooled model.
    """
    d = np.asarray(p_seg, dtype="float64") - np.asarray(p_pool, dtype="float64")
    r = np.asarray(y, dtype="float64") - np.asarray(p_pool, dtype="float64")
    denom = float(np.dot(d, d))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    return float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--inner-val-months", type=int, default=1,
                    help="how many of the LAST training months are held back to "
                         "solve the blend weights on")
    ap.add_argument("--lgb-config", default="base_lr05_lv255_md100",
                    choices=sorted(LGB_GRID),
                    help="LightGBM configuration for BOTH the pooled and the "
                         "per-airport halves. Default reproduces the original run. "
                         "`per_airport_tuned` measured that the gain from tuning is "
                         "a better GLOBAL config (-3.368 s val) and that choosing "
                         "one per airport adds only 0.118 s against a 0.29 s seed "
                         "spread -- so this is deliberately ONE config, not ten.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--freeze-from", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    Xtr, ytr, pool_meta = D.load_part(args.split, "train", args.featureset,
                                      filter_target=True,
                                      outlier_policy=args.outlier_policy)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]
    ytr_a = np.asarray(ytr, dtype="float64")

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set. Pass --freeze-from.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit("--freeze-from points at a non-selection surface.")
        Xva = yva = meta_va = None
    else:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset,
                                        filter_target=False)
        Xva = D.align_columns(Xtr, Xva)

    params = dict(POOLED_PARAMS)
    params.update(LGB_GRID[args.lgb_config])
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed, "device_type": args.device})

    months = pool_meta["month"].astype(str)
    order = sorted(months.unique())
    print("[{}] train n={} months={} -> {}".format(ARM, len(Xtr), len(order), order))

    # ---------------- blend weights, from an inner time-based holdout ----------
    if frozen:
        alphas = {str(k): float(v) for k, v in frozen["alpha_per_airport"].items()}
        inner = frozen.get("inner_holdout")
        print("[{}] weights frozen from {}".format(ARM, args.freeze_from))
    else:
        if len(order) <= args.inner_val_months:
            raise SystemExit(
                "training pool has {} month(s); cannot hold back {} for the inner "
                "split".format(len(order), args.inner_val_months))
        inner_val_months = order[-args.inner_val_months:]
        m_in = ~months.isin(inner_val_months).to_numpy()
        m_iv = ~m_in
        print("[{}] inner split: fit on {} ({} rows), solve weights on {} ({} rows)"
              .format(ARM, order[:-args.inner_val_months], int(m_in.sum()),
                      inner_val_months, int(m_iv.sum())))

        Xi, yi, mi = Xtr[m_in], ytr_a[m_in], pool_meta[m_in]
        Xv2, yv2, mv2 = Xtr[m_iv], ytr_a[m_iv], pool_meta[m_iv]

        # pooled model on the inner-train months
        dtr = lgb.Dataset(Xi, label=yi, categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(Xv2, label=yv2, categorical_feature=cats, reference=dtr,
                          free_raw_data=False)
        pooled_inner = lgb.train(params, dtr, num_boost_round=args.num_boost_round,
                                 valid_sets=[dva], valid_names=["val"],
                                 callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                            lgb.log_evaluation(0)])
        p_pool = pooled_inner.predict(Xv2, num_iteration=pooled_inner.best_iteration)

        # per-airport model on the same inner-train months
        seg_inner = PerAirportLGBM(params, args.num_boost_round, args.patience, cats)
        seg_inner.fit(Xi, yi, mi, Xv2, yv2, mv2, lgb=lgb, verbose=False)
        p_seg = seg_inner.predict(Xv2, mv2)

        ap_iv = mv2[AIRPORT_COL].astype("object").to_numpy()
        alphas, diag = {}, []
        for a in sorted(pd.unique(ap_iv)):
            m = ap_iv == a
            al = solve_alpha(yv2[m], p_seg[m], p_pool[m])
            alphas[str(a)] = al
            blend = al * p_seg[m] + (1 - al) * p_pool[m]
            diag.append({"airport": str(a), "n_inner_val": int(m.sum()), "alpha": al,
                         "inner_rmse_pooled": rmse(yv2[m], p_pool[m]),
                         "inner_rmse_per_airport": rmse(yv2[m], p_seg[m]),
                         "inner_rmse_blend": rmse(yv2[m], blend)})
            print("   {:6s} n={:6d} alpha={:.3f}  inner RMSE pooled={:7.2f} "
                  "seg={:7.2f} blend={:7.2f}".format(
                      str(a), int(m.sum()), al, diag[-1]["inner_rmse_pooled"],
                      diag[-1]["inner_rmse_per_airport"], diag[-1]["inner_rmse_blend"]))
        inner = {"inner_val_months": inner_val_months,
                 "n_inner_train": int(m_in.sum()), "n_inner_val": int(m_iv.sum()),
                 "per_airport": diag}

    # ---------------- final models, on the FULL training pool ------------------
    if refit_only:
        ratio = len(Xtr) / max(1, frozen["n_train"])
        n_pool_rounds = max(1, int(round(frozen["pooled_best_iteration"] * ratio)))
        print("  frozen pooled rounds {} x pool ratio {:.3f} -> {}".format(
            frozen["pooled_best_iteration"], ratio, n_pool_rounds))
        dtr = lgb.Dataset(Xtr, label=ytr_a, categorical_feature=cats, free_raw_data=False)
        pooled = lgb.train(params, dtr, num_boost_round=n_pool_rounds,
                           callbacks=[lgb.log_evaluation(0)])
        pooled.best_iteration = n_pool_rounds
        seg = PerAirportLGBM(params, args.num_boost_round, args.patience, cats)
        seg.fit(Xtr, ytr_a, pool_meta, lgb=lgb, frozen=frozen, verbose=False)
        val_rmse_clipped = None
    else:
        dtr = lgb.Dataset(Xtr, label=ytr_a, categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(Xva, label=np.asarray(yva, dtype="float64"),
                          categorical_feature=cats, reference=dtr, free_raw_data=False)
        pooled = lgb.train(params, dtr, num_boost_round=args.num_boost_round,
                           valid_sets=[dva], valid_names=["val"],
                           callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                      lgb.log_evaluation(0)])
        seg = PerAirportLGBM(params, args.num_boost_round, args.patience, cats)
        seg.fit(Xtr, ytr_a, pool_meta, Xva, yva, meta_va, lgb=lgb, verbose=False)

        vp_pool = pooled.predict(Xva, num_iteration=pooled.best_iteration)
        vp_seg = seg.predict(Xva, meta_va)
        ap_v = meta_va[AIRPORT_COL].astype("object").to_numpy()
        blend = np.empty(len(Xva), dtype="float64")
        for a in pd.unique(ap_v):
            m = ap_v == a
            al = alphas.get(str(a), 0.0)
            blend[m] = al * vp_seg[m] + (1 - al) * vp_pool[m]
        vp = np.clip(blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, vp)
        print("[{}] val RMSE (clipped) = {:.3f}   [pooled alone {:.3f} | "
              "per-airport alone {:.3f}]".format(
                  ARM, val_rmse_clipped,
                  rmse(yva, np.clip(vp_pool, config.PREDICTION_CLIP_MIN_SEC,
                                    config.PREDICTION_CLIP_MAX_SEC)),
                  rmse(yva, np.clip(vp_seg, config.PREDICTION_CLIP_MIN_SEC,
                                    config.PREDICTION_CLIP_MAX_SEC))))

    sel = declare_selection_set(
        "per_airport_stack blend weights",
        selected_on="frozen" if refit_only else "inner_holdout_from_train",
        value=alphas)

    with open(out / "model.pkl", "wb") as fh:
        pickle.dump({"pooled": pooled, "seg": seg, "alphas": alphas}, fh)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "alpha_per_airport": alphas,
        "alpha_note": ("1.0 = the per-airport model alone, 0.0 = the pooled model "
                       "alone; solved per airport by least squares on an inner "
                       "time-based holdout carved from TRAIN, never on val"),
        "inner_holdout": inner,
        "lgb_config": args.lgb_config,
        "pooled_best_iteration": int(pooled.best_iteration),
        "best_iteration_per_airport": {k: int(v) for k, v in seg.best_iter_.items()},
        "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
