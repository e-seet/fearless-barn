#!/usr/bin/env python3
"""Arm lightgbm_congest: the pooled LightGBM reference plus the realised-taxi
congestion columns.

Why this arm exists
-------------------
`pipeline_per_airport_congest` showed the congestion columns are worth about
-1.6 s of corrected val on the PER-AIRPORT tree arm, and that arm is the only
place they have ever been tested. That leaves a question the adoption decision
did not answer: is the signal general, or is it something a per-airport model
can use and a pooled one cannot? The two are not the same finding. The columns
are absolute seconds, and the airports they come from run at very different
speeds (LIRF ~700 s against LSZH ~160 s), so a pooled tree has to spend splits
conditioning on airport before the column means anything, while a per-airport
model gets it pre-conditioned for free. A null here would narrow the claim, not
refute it.

Structure is copied from `pipeline_lightgbm_arrfeat` deliberately: same pool,
outlier policy, PARAMS, early stopping and artifact keying as the `lightgbm`
reference arm, so the ONLY difference between this and the untagged `lightgbm`
runs it is paired against is which columns get joined. `attach` is imported from
`pipeline_per_airport_congest.extra` rather than reimplemented, so both arms
provably read the same cache, keyed on the same code hash.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
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
from pipeline_lightgbm.train import PARAMS
from pipeline_per_airport_congest.extra import (ADDED_COLUMNS, attach,
                                                code_fingerprint)

ARM = "lightgbm_congest"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--num-leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--min-data-in-leaf", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--learning-rate", type=float, default=PARAMS["learning_rate"])
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
    Xtr = attach(Xtr, "train2025", args.featureset)
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
    else:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset,
                                  filter_target=False)
        Xva = D.align_columns(Xtr, attach(Xva, "train2025", args.featureset))

    params = dict(frozen["params"]) if frozen else dict(PARAMS)
    if not frozen:
        params.update({"learning_rate": args.learning_rate,
                       "num_leaves": args.num_leaves,
                       "min_data_in_leaf": args.min_data_in_leaf})
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed, "device_type": args.device})

    print("[{}] train n={} feats={} (+{} extra) cats={}".format(
        ARM, len(Xtr), Xtr.shape[1], len(ADDED_COLUMNS), len(cats)))

    dtr = lgb.Dataset(Xtr, label=ytr_a, categorical_feature=cats, free_raw_data=False)
    if refit_only:
        ratio = len(Xtr) / max(1, frozen["n_train"])
        rounds = max(1, int(round(frozen["best_iteration"] * ratio)))
        print("  frozen best_iteration {} x pool ratio {:.3f} -> {} rounds "
              "(heuristic, not a measured optimum)".format(
                  frozen["best_iteration"], ratio, rounds))
        booster = lgb.train(params, dtr, num_boost_round=rounds,
                            callbacks=[lgb.log_evaluation(0)])
        booster.best_iteration = rounds
        conv = {"refit_only": True, "hit_max_epochs": False, "rounds": rounds,
                "frozen_from": args.freeze_from,
                "note": "no early stopping: this protocol has no val by design"}
        val_rmse_clipped = None
    else:
        dva = lgb.Dataset(Xva, label=np.asarray(yva, dtype="float64"),
                          categorical_feature=cats, reference=dtr, free_raw_data=False)
        hist = {}
        booster = lgb.train(params, dtr, num_boost_round=args.num_boost_round,
                            valid_sets=[dva], valid_names=["val"],
                            callbacks=[lgb.early_stopping(args.patience, verbose=False),
                                       lgb.log_evaluation(0),
                                       lgb.record_evaluation(hist)])
        curve = hist["val"]["rmse"]
        conv = convergence_evidence(
            [{"epoch": i + 1, "val_rmse": v} for i, v in enumerate(curve)],
            max_epochs=args.num_boost_round, patience=args.patience,
            metric_key="val_rmse", mode="min")
        vp = np.clip(booster.predict(Xva, num_iteration=booster.best_iteration),
                     config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, vp)
        print("[{}] val RMSE (clipped) = {:.3f}  (best_iter={}, ES fired={})".format(
            ARM, val_rmse_clipped, booster.best_iteration,
            not conv.get("hit_max_epochs", True)))

    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain"),
                        "split": booster.feature_importance("split")})
    imp.sort_values("gain", ascending=False).to_parquet(
        out / "feature_importance.parquet", index=False)
    # Recorded because a null result has two very different explanations: the
    # model ignored the columns, or it used them and they did not pay.
    share = float(imp.loc[imp.feature.isin(list(ADDED_COLUMNS)), "gain"].sum()
                  / max(1e-9, imp.gain.sum()))
    ranks = {r.feature: int(i + 1) for i, r in enumerate(
        imp.sort_values("gain", ascending=False).itertuples())
        if r.feature in ADDED_COLUMNS}
    print("  extra columns take {:.1%} of total gain; ranks: {}".format(
        share, dict(sorted(ranks.items(), key=lambda kv: kv[1]))))

    sel = declare_selection_set("lightgbm_congest best_iteration",
                                selected_on="frozen" if refit_only else "val",
                                value=int(booster.best_iteration))
    booster.save_model(str(out / "model.txt"), num_iteration=booster.best_iteration)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "extra_features": {"columns": list(ADDED_COLUMNS),
                           "code_fingerprint": code_fingerprint(),
                           "gain_share": share, "gain_rank": ranks,
                           "uses_dep_target": False},
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration": int(booster.best_iteration),
        "val_rmse_clipped": val_rmse_clipped, "refit_only": bool(refit_only),
        "convergence": conv, "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
