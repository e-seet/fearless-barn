#!/usr/bin/env python3
"""Arm 3 (learned, expected winner): LightGBM with native categorical handling.

Why LightGBM over the alternatives here: 1.4M rows x ~50 features with several
high-cardinality categoricals (6,345 stand-runway pairs) and a strongly
non-linear, interaction-heavy target. Native categorical splits avoid both the
one-hot blow-up and the target-encoding leak surface the ridge arm has to manage.

Objective is L2 (= the leaderboard metric) so early stopping, the objective and
the score all agree. Convergence is by early stopping on VAL RMSE, never by the
round cap -- `num_boost_round` is a safety ceiling and hitting it is reported as
a finding, not accepted as a result.
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

ARM = "lightgbm"

PARAMS = {
    "objective": "l2",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 255,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "verbosity": -1,
    "num_threads": 0,
}


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
    ap.add_argument("--num-leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--min-data-in-leaf", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"],
                    help="LightGBM device_type; 'cuda' needs a CUDA-built LightGBM")
    ap.add_argument("--freeze-from", default=None,
                    help="path to an honest run's model.json. Required for the "
                         "no-val 'full' refit protocol: params AND the round "
                         "count arrive frozen, because best_iteration cannot be "
                         "re-derived without a validation set.")
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES),
                    help="how the TRAINING tail is handled; part of the artifact "
                         "key. Evaluation sets are never filtered.")
    ap.add_argument("--tag", default=None,
                    help="OPTIONAL label appended to the artifact key, for runs that "
                         "vary something the protocol key does not name -- a "
                         "hyperparameter sweep, a loss variant. Without it such a run "
                         "silently overwrites the protocol's registered run "
                         "(MISTAKES.md #35/#50).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                 filter_target=True,
                                 outlier_policy=args.outlier_policy)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]

    # ---- the no-val refit path (design-gate finding B4) ----
    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set, so nothing can be selected on "
                "it. Pass --freeze-from <honest run>/model.json to supply the "
                "hyperparameters and the round count.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit(
                "--freeze-from points at a run on split '{}', which is not a valid "
                "selection surface. Freeze from a 'forward' run.".format(frozen["split"]))
        print("[{}] REFIT on '{}' with hyperparameters frozen from {} "
              "(split={}, featureset={}, best_iteration={})".format(
                  ARM, args.split, args.freeze_from, frozen["split"],
                  frozen["featureset"], frozen["best_iteration"]))
    else:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
    print("[{}] train n={} val n={} feats={} cats={} device={}".format(
        ARM, len(Xtr), 0 if refit_only else len(Xva), Xtr.shape[1], len(cats),
        args.device))

    if frozen:
        params = dict(frozen["params"])
        params.update({"seed": args.seed, "bagging_seed": args.seed,
                       "feature_fraction_seed": args.seed, "device_type": args.device})
    else:
        params = dict(PARAMS)
        params.update({"learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
                       "min_data_in_leaf": args.min_data_in_leaf, "seed": args.seed,
                       "bagging_seed": args.seed, "feature_fraction_seed": args.seed,
                       "device_type": args.device})

    dtr = lgb.Dataset(Xtr, label=np.asarray(ytr, dtype="float64"),
                      categorical_feature=cats, free_raw_data=False)

    if refit_only:
        # best_iteration cannot be re-derived without a val set. Scale the frozen
        # count by the pool-size ratio: the refit pool is larger, so at the same
        # learning rate it supports proportionally more rounds. This is a stated
        # heuristic, not a measured optimum -- recorded as such in model.json.
        ratio = len(Xtr) / max(1, frozen["n_train"])
        n_rounds = max(1, int(round(frozen["best_iteration"] * ratio)))
        print("  frozen best_iteration {} x pool ratio {:.3f} -> {} rounds "
              "(heuristic, not a measured optimum)".format(
                  frozen["best_iteration"], ratio, n_rounds))
        history = {}
        booster = lgb.train(params, dtr, num_boost_round=n_rounds,
                            callbacks=[lgb.log_evaluation(100)])
        booster.best_iteration = n_rounds
        conv = {"refit_only": True, "hit_max_epochs": False,
                "rounds": n_rounds, "frozen_from": args.freeze_from,
                "frozen_best_iteration": frozen["best_iteration"],
                "pool_ratio": ratio,
                "note": "no early stopping: this protocol has no validation set "
                        "by design; the round count is frozen, not selected"}
        val_curve = []
    else:
        dva = lgb.Dataset(Xva, label=np.asarray(yva, dtype="float64"),
                          categorical_feature=cats, reference=dtr, free_raw_data=False)
        history = {}
        booster = lgb.train(
            params, dtr, num_boost_round=args.num_boost_round,
            valid_sets=[dtr, dva], valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(args.patience, verbose=True),
                       lgb.log_evaluation(100),
                       lgb.record_evaluation(history)],
        )

    # ---- convergence evidence: early stopping must have FIRED, not run out ----
    if not refit_only:
        val_curve = history["val"]["rmse"]
        hist = [{"epoch": i + 1, "val_rmse": v} for i, v in enumerate(val_curve)]
        conv = convergence_evidence(hist, max_epochs=args.num_boost_round,
                                    patience=args.patience, metric_key="val_rmse",
                                    mode="min")
        print("  best_iteration={} of cap {} | early stopping fired: {}".format(
            booster.best_iteration, args.num_boost_round,
            not conv.get("hit_max_epochs", True)))
        if conv.get("hit_max_epochs"):
            print("  WARNING: hit the round cap -- this is a signal to raise the cap, "
                  "not a converged result. Do not register this number.")

    sel = declare_selection_set(
        "lightgbm best_iteration",
        selected_on="frozen" if refit_only else "val",
        value=int(booster.best_iteration))
    booster.save_model(str(out / "model.txt"), num_iteration=booster.best_iteration)
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain"),
                        "split": booster.feature_importance("split")})
    imp.sort_values("gain", ascending=False).to_parquet(out / "feature_importance.parquet",
                                                        index=False)
    if refit_only:
        val_rmse_raw = val_rmse_clipped = None
    else:
        val_pred = booster.predict(Xva, num_iteration=booster.best_iteration)
        val_pred = np.clip(val_pred, config.PREDICTION_CLIP_MIN_SEC,
                           config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_raw = float(val_curve[booster.best_iteration - 1])
        val_rmse_clipped = rmse(yva, val_pred)

    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration": int(booster.best_iteration),
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
    print("  top-15 features by gain:")
    for r in imp.sort_values("gain", ascending=False).head(15).itertuples():
        print("    {:28s} gain={:14.0f} splits={:7d}".format(r.feature, r.gain, r.split))
    return 0


if __name__ == "__main__":
    sys.exit(main())
