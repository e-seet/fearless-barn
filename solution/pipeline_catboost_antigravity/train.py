#!/usr/bin/env python3
"""Arm catboost_antigravity: CatBoost with ordered-target-statistics categoricals.

Motivated by real competitor evidence, not a guess: team zestful-fountain
(rank #16/131 at 273.56 RMSE on the live DC2026 leaderboard, ahead of our
295.21) uses CatBoost as their primary model. Structurally different from
both existing tree arms here: `lightgbm` (histogram, leaf-wise) and
`xgboost_antigravity` (histogram, level-wise) both split categoricals by
grouping bins; CatBoost's ordered target statistics encode each categorical
level by (leakage-safe, permutation-ordered) target mean instead, and its
trees are symmetric/oblivious rather than leaf-wise. The standalone bar to
clear is LOW -- our own single-model LightGBM (val_rmse_clipped ~296-298)
already loses to the MLP (~254-261) -- so the real question this arm exists
to answer is not "does CatBoost win alone" but "does it make different
enough errors from the MLP to be worth a nonzero blend weight."

Mirrors pipeline_lightgbm/train.py's structure (CLI args, D.load_part data
path, categorical list, early-stopping/provenance conventions) deliberately,
so this arm slots into the existing comparison/blending machinery without
special-casing.
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
from common import features as F
from common.metrics import rmse
from common.bootstrap import (convergence_evidence, declare_selection_set, run_dir,
                              seed_everything, write_json_guarded)

ARM = "catboost_antigravity"

PARAMS = {
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bayesian",
}


def _cat_frame(X: "pd.DataFrame", cats: list[str]) -> "pd.DataFrame":
    """CatBoost wants plain strings for categoricals, NaN as an explicit
    sentinel category rather than a float NaN CatBoost has to special-case --
    matches how every other arm here treats "unknown" as its own bucket
    (e.g. mlp_antigravity's unk = len(cat_maps[col]))."""
    Xc = X.copy()
    for c in cats:
        Xc[c] = Xc[c].astype("object").fillna("__NA__").astype(str)
    return Xc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000,
                    help="SAFETY CAP, not the stopping condition (see --patience)")
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--depth", type=int, default=PARAMS["depth"])
    ap.add_argument("--l2-leaf-reg", type=float, default=PARAMS["l2_leaf_reg"])
    ap.add_argument("--device", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--freeze-from", default=None,
                    help="path to an honest run's model.json. Required for the "
                         "no-val 'full' refit protocol.")
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from catboost import CatBoostRegressor, Pool

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                       filter_target=True,
                                       outlier_policy=args.outlier_policy)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set. Pass --freeze-from "
                "<honest run>/model.json.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit(
                "--freeze-from points at split '{}', not a valid selection "
                "surface. Freeze from a 'forward' run.".format(frozen["split"]))
        print("[{}] REFIT on '{}' with hyperparameters frozen from {} "
              "(best_iteration={})".format(ARM, args.split, args.freeze_from,
                                            frozen["best_iteration"]))
    else:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)

    print("[{}] train n={} val n={} feats={} cats={} device={}".format(
        ARM, len(Xtr), 0 if refit_only else len(Xva), Xtr.shape[1], len(cats), args.device))

    Xtr_c = _cat_frame(Xtr, cats)
    if not refit_only:
        Xva_c = _cat_frame(Xva, cats)

    if frozen:
        params = dict(frozen["params"])
        params.update({"random_seed": args.seed})
    else:
        params = dict(PARAMS)
        params.update({"learning_rate": args.learning_rate, "depth": args.depth,
                       "l2_leaf_reg": args.l2_leaf_reg, "random_seed": args.seed})
    params["task_type"] = "GPU" if args.device == "gpu" else "CPU"
    if args.device == "gpu":
        params["devices"] = "0"

    dtr = Pool(Xtr_c, label=np.asarray(ytr, dtype="float64"), cat_features=cats)

    if refit_only:
        ratio = len(Xtr) / max(1, frozen["n_train"])
        n_rounds = max(1, int(round(frozen["best_iteration"] * ratio)))
        print("  frozen best_iteration {} x pool ratio {:.3f} -> {} rounds "
              "(heuristic, not a measured optimum)".format(
                  frozen["best_iteration"], ratio, n_rounds))
        model = CatBoostRegressor(iterations=n_rounds, **params, verbose=100)
        model.fit(dtr)
        best_iteration = n_rounds
        conv = {"refit_only": True, "hit_max_epochs": False,
                "rounds": n_rounds, "frozen_from": args.freeze_from,
                "frozen_best_iteration": frozen["best_iteration"],
                "pool_ratio": ratio,
                "note": "no early stopping: this protocol has no validation set "
                        "by design; the round count is frozen, not selected"}
        val_curve = []
    else:
        dva = Pool(Xva_c, label=np.asarray(yva, dtype="float64"), cat_features=cats)
        model = CatBoostRegressor(iterations=args.num_boost_round, **params,
                                  early_stopping_rounds=args.patience, verbose=100)
        model.fit(dtr, eval_set=dva, use_best_model=True)
        best_iteration = model.get_best_iteration()
        val_curve = list(model.get_evals_result()["validation"]["RMSE"])
        hist = [{"epoch": i + 1, "val_rmse": v} for i, v in enumerate(val_curve)]
        conv = convergence_evidence(hist, max_epochs=args.num_boost_round,
                                    patience=args.patience, metric_key="val_rmse",
                                    mode="min")
        print("  best_iteration={} of cap {} | early stopping fired: {}".format(
            best_iteration, args.num_boost_round, not conv.get("hit_max_epochs", True)))
        if conv.get("hit_max_epochs"):
            print("  WARNING: hit the round cap -- raise the cap, do not register this.")

    sel = declare_selection_set(
        "catboost_antigravity best_iteration",
        selected_on="frozen" if refit_only else "val",
        value=int(best_iteration))
    model.save_model(str(out / "model.cbm"))
    imp = pd.DataFrame({"feature": Xtr_c.columns,
                        "importance": model.get_feature_importance(dtr)})
    imp.sort_values("importance", ascending=False).to_parquet(
        out / "feature_importance.parquet", index=False)

    if refit_only:
        val_rmse_raw = val_rmse_clipped = None
    else:
        val_pred = model.predict(Xva_c)
        val_pred = np.clip(val_pred, config.PREDICTION_CLIP_MIN_SEC,
                           config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_raw = float(val_curve[best_iteration])
        val_rmse_clipped = rmse(yva, val_pred)

    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration": int(best_iteration),
        "val_rmse_raw": val_rmse_raw, "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "convergence": conv, "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(_pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)

    print("[{}] {} -> {}".format(
        ARM,
        "refit complete (no val by design)" if refit_only
        else "val RMSE (clipped) = {:.3f}".format(val_rmse_clipped), out))
    print("  top-15 features by importance:")
    for r in imp.sort_values("importance", ascending=False).head(15).itertuples():
        print("    {:28s} importance={:8.3f}".format(r.feature, r.importance))
    return 0


if __name__ == "__main__":
    sys.exit(main())
