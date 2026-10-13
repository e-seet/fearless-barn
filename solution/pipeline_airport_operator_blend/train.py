#!/usr/bin/env python3
"""Arm airport_operator_blend: convex blend of per_airport and per_operator.

Where this comes from
----------------------
Both per_airport and per_operator segment the SAME rows by different columns,
and neither has ever been combined with the other in this build. An isolated
scratch check (not this code path) found the two agree well enough alone
(val ~278.2 and ~277.8) but disagree enough row-by-row that a plain 0.5/0.5
average of their predictions scores ~271.9 -- beating either alone, and beating
every other tree-based arm in this build including per_airport_tuned (275.49).
This arm formalises that finding with the weight solved honestly instead of
eyeballed at 0.5.

One GLOBAL weight, not one per airport
---------------------------------------
per_airport_stack (pooled + per_airport) solves one alpha PER AIRPORT, because
the two models it blends disagree in an airport-dependent way (LIRF wants the
segmented model, most others lean pooled). The scratch check behind THIS arm
never found that kind of per-airport structure -- the two models here are both
already airport-aware in different ways (one splits on it directly, the other
never sees it and instead splits on operator), so there is no prior reason to
expect the right mix to vary by airport on top of that. One global weight is
also the simpler, lower-variance hypothesis, and MISTAKES.md's own lesson from
this same investigation (a 9-cell per-airport x-3-arm weight grid measured
246.94 in-sample and 254.24 -- WORSE than either base arm -- once solved
honestly out-of-sample) is a direct warning against fitting more free
parameters than the data actually supports. Kept as a single scalar until a
per-airport version is measured to actually help.

Why the weight is NOT fitted on val
-------------------------------------
Same reasoning as per_airport_stack: an inner, time-based holdout is carved
from the END of the training pool (the last `--inner-val-months` months), both
models are fit on the months before it, and the weight is solved on that held
month. The final models are then refit on the FULL training pool and the
already-solved weight is applied at val/test time -- so val is never touched
by anything free-parameter-fitted here, and compare.py can rank this arm
against the others at face value.
"""
from __future__ import annotations

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
from pipeline_per_airport.model import PerAirportLGBM
from pipeline_per_operator.model import PerOperatorLGBM, DEFAULT_TOP_N

ARM = "airport_operator_blend"

PARAMS = {
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}


def solve_alpha(y, p_a, p_b) -> float:
    """Least-squares GLOBAL weight on model a, clipped to [0, 1].

    minimise  sum (y - (a*p_a + (1-a)*p_b))^2  -- same derivation as
    per_airport_stack.solve_alpha, just not run separately per airport.
    """
    d = np.asarray(p_a, dtype="float64") - np.asarray(p_b, dtype="float64")
    r = np.asarray(y, dtype="float64") - np.asarray(p_b, dtype="float64")
    denom = float(np.dot(d, d))
    if denom <= 0 or not np.isfinite(denom):
        return 0.5
    return float(np.clip(np.dot(d, r) / denom, 0.0, 1.0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--inner-val-months", type=int, default=1,
                    help="how many of the LAST training months are held back to "
                         "solve the blend weight on")
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--congest", action="store_true",
                    help="attach the realised-taxi congestion columns "
                         "(pipeline_per_airport_congest/extra.py). Off by "
                         "default so the tier-A arm's behaviour is unchanged "
                         "unless asked for. Recorded in model.json so evaluate "
                         "and predict reproduce the same feature frame.")
    ap.add_argument("--adsb", action="store_true",
                    help="attach the all_nd adsb.lol ADS-B columns (tests_adsb_full/attach.py; "
                         "PREREGISTERED_tree.md). Off by default; recorded in model.pkl/json.")
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
    if args.congest:
        from pipeline_per_airport_congest.extra import attach as attach_congest
        Xtr = attach_congest(Xtr, "train2025", args.featureset)
    if args.adsb:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests_adsb_full"))
        from attach import attach as attach_adsb
        Xtr = attach_adsb(Xtr)
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
        if args.congest:
            Xva = attach_congest(Xva, "train2025", args.featureset)
        if args.adsb:
            Xva = attach_adsb(Xva)
        Xva = D.align_columns(Xtr, Xva)

    params = dict(PARAMS)
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed, "device_type": args.device})

    months = pool_meta["month"].astype(str)
    order = sorted(months.unique())
    print("[{}] train n={} months={} -> {}".format(ARM, len(Xtr), len(order), order))

    # ---------------- blend weight, from an inner time-based holdout -----------
    if frozen:
        alpha = float(frozen["alpha"])
        vocab = list(frozen["operator_vocab"])
        inner = frozen.get("inner_holdout")
        print("[{}] weight frozen from {}: alpha={:.3f}".format(ARM, args.freeze_from, alpha))
    else:
        if len(order) <= args.inner_val_months:
            raise SystemExit(
                "training pool has {} month(s); cannot hold back {} for the inner "
                "split".format(len(order), args.inner_val_months))
        inner_val_months = order[-args.inner_val_months:]
        m_in = ~months.isin(inner_val_months).to_numpy()
        m_iv = ~m_in
        print("[{}] inner split: fit on {} ({} rows), solve weight on {} ({} rows)"
              .format(ARM, order[:-args.inner_val_months], int(m_in.sum()),
                      inner_val_months, int(m_iv.sum())))

        Xi, yi, mi = Xtr[m_in], ytr_a[m_in], pool_meta[m_in]
        Xv2, yv2, mv2 = Xtr[m_iv], ytr_a[m_iv], pool_meta[m_iv]

        seg_a_inner = PerAirportLGBM(params, args.num_boost_round, args.patience, cats)
        seg_a_inner.fit(Xi, yi, mi, Xv2, yv2, mv2, lgb=lgb, verbose=False)
        p_a = seg_a_inner.predict(Xv2, mv2)

        seg_b_inner = PerOperatorLGBM(params, args.num_boost_round, args.patience,
                                      cats, top_n=args.top_n)
        seg_b_inner.fit(Xi, yi, Xi, Xv2, yv2, Xv2, lgb=lgb, verbose=False)
        vocab = seg_b_inner.vocab_
        p_b = seg_b_inner.predict(Xv2, Xv2)

        alpha = solve_alpha(yv2, p_a, p_b)
        inner_rmse_a = rmse(yv2, p_a)
        inner_rmse_b = rmse(yv2, p_b)
        inner_rmse_blend = rmse(yv2, alpha * p_a + (1 - alpha) * p_b)
        print("  alpha (weight on per_airport) = {:.3f}  inner RMSE airport={:.2f} "
              "operator={:.2f} blend={:.2f}".format(
                  alpha, inner_rmse_a, inner_rmse_b, inner_rmse_blend))
        inner = {"inner_val_months": inner_val_months,
                 "n_inner_train": int(m_in.sum()), "n_inner_val": int(m_iv.sum()),
                 "inner_rmse_per_airport": inner_rmse_a,
                 "inner_rmse_per_operator": inner_rmse_b,
                 "inner_rmse_blend": inner_rmse_blend}

    # ---------------- final models, on the FULL training pool ------------------
    seg_a = PerAirportLGBM(params, args.num_boost_round, args.patience, cats)
    seg_b = PerOperatorLGBM(params, args.num_boost_round, args.patience, cats,
                            top_n=args.top_n)
    if refit_only:
        # The refit pool has no val set, so BOTH segments must receive their
        # round counts frozen from the honest run or LightGBM silently trains to
        # --num-boost-round (5000) with no early stopping.
        #
        # This was broken: the code read `frozen["per_airport_frozen"]` and
        # passed nothing at all for the operator side, but the writer below
        # stores those counts as `best_iteration_per_airport` /
        # `best_iteration_per_operator`. `.get()` on the wrong key returned None,
        # `frozen=None` is the "no frozen payload" signal, and the refit quietly
        # trained all 31 boosters (10 airports + 21 operators) to 5000 rounds
        # instead of the ~100-190 the honest run selected -- ~30x over-training,
        # with nothing in the output saying so. The standalone per_airport and
        # per_operator arms hand their model classes the whole model.json, which
        # is the shape those classes actually index; this now does the same, with
        # the operator vocab renamed to the key PerOperatorLGBM expects.
        missing = [k for k in ("best_iteration_per_airport", "best_iteration_per_operator")
                   if not frozen.get(k)]
        if missing:
            raise SystemExit(
                "--freeze-from run {} has no {}. A no-val refit cannot pick a "
                "round count, so proceeding would train every booster to the "
                "{}-round cap. Re-run the 'forward' arm to regenerate its "
                "model.json.".format(args.freeze_from, " and ".join(missing),
                                     args.num_boost_round))
        seg_a.fit(Xtr, ytr_a, pool_meta, lgb=lgb,
                  frozen={"best_iteration_per_airport": frozen["best_iteration_per_airport"]},
                  verbose=False)
        seg_b.fit(Xtr, ytr_a, Xtr, lgb=lgb,
                  frozen={"vocab": vocab,
                          "best_iteration_per_operator": frozen["best_iteration_per_operator"]},
                  verbose=False)
        val_rmse_clipped = None
    else:
        seg_a.fit(Xtr, ytr_a, pool_meta, Xva, yva, meta_va, lgb=lgb, verbose=False)
        seg_b.fit(Xtr, ytr_a, Xtr, Xva, yva, Xva, lgb=lgb, vocab=vocab, verbose=False)

        vp_a = seg_a.predict(Xva, meta_va)
        vp_b = seg_b.predict(Xva, Xva)
        blend = alpha * vp_a + (1 - alpha) * vp_b
        vp = np.clip(blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, vp)
        print("[{}] val RMSE (clipped) = {:.3f}   [per_airport alone {:.3f} | "
              "per_operator alone {:.3f}]".format(
                  ARM, val_rmse_clipped,
                  rmse(yva, np.clip(vp_a, config.PREDICTION_CLIP_MIN_SEC,
                                    config.PREDICTION_CLIP_MAX_SEC)),
                  rmse(yva, np.clip(vp_b, config.PREDICTION_CLIP_MIN_SEC,
                                    config.PREDICTION_CLIP_MAX_SEC))))

    sel = declare_selection_set(
        "airport_operator_blend global alpha",
        selected_on="frozen" if refit_only else "inner_holdout_from_train",
        value=alpha)

    with open(out / "model.pkl", "wb") as fh:
        pickle.dump({"per_airport": seg_a, "per_operator": seg_b, "congest": args.congest, "adsb": args.adsb,
        "alpha": alpha}, fh)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params, "top_n": args.top_n,
        "congest": args.congest,
        "adsb": args.adsb,
        "alpha": alpha,
        "alpha_note": ("1.0 = per_airport alone, 0.0 = per_operator alone; ONE "
                       "global weight (not per-airport -- see module docstring), "
                       "solved by least squares on an inner time-based holdout "
                       "carved from TRAIN, never on val"),
        "operator_vocab": vocab,
        "inner_holdout": inner,
        "best_iteration_per_airport": {k: int(v) for k, v in seg_a.best_iter_.items()},
        "best_iteration_per_operator": {k: int(v) for k, v in seg_b.best_iter_.items()},
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
