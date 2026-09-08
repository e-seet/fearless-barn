#!/usr/bin/env python3
"""Arm lightgbm_unsup: LightGBM plus unsupervised-learned structure features.

The hypothesis
--------------
Two things this dataset withholds are recoverable without labels:

* **Airport layout.** `PLAN.md` sec.9 records "airport layout / stand coordinates"
  as an open lever that could not be taken, because the release ships only
  categorical stand and runway identifiers and no coordinates. But taxi distance
  is the physical driver of taxi-out, and layout leaves an observable trace: the
  stand x runway co-occurrence matrix says which aprons feed which runways.
  Factorising it (truncated SVD on log1p counts) gives every stand a dense vector
  whose neighbours are stands that route the same way -- a learned stand-in for
  the geometry, built with no target.
* **Day regimes.** A disrupted day is slow for everything at that airport, and no
  per-row feature says so. Clustering per-(airport, day) traffic profiles gives a
  small categorical for that, again with no target.

It also targets a concrete weakness of the native-categorical approach:
`stand_runway` carries 6,345 levels, and a rare level gives a categorical split
almost nothing to work with, while its SVD coordinates are informed by every
other movement through the same stand.

Discipline
----------
Both transforms are fitted on the **training pool only** and applied unchanged to
val / test / ranking -- the same fit-after-split rule the supervised
preprocessing follows. Nothing here reads the target at any point, so there is no
target-encoding surface: the label-permutation control cannot detect a leak that
does not exist, and the honest statement is that the leak is closed by
construction rather than by a passing control.

Everything else -- features, pool, outlier policy, early stopping, artifact
keying -- is identical to the `lightgbm` arm, so the comparison isolates the
unsupervised features.
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
from common.bootstrap import (convergence_evidence, declare_selection_set, run_dir,
                              seed_everything, write_json_guarded)
from pipeline_lightgbm.train import PARAMS
from pipeline_lightgbm_unsup.unsup import UnsupervisedFeatures

ARM = "lightgbm_unsup"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--svd-components", type=int, default=4)
    ap.add_argument("--day-clusters", type=int, default=6)
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
    ytr_a = np.asarray(ytr, dtype="float64")

    # ---- unsupervised transforms: FIT ON TRAIN ONLY ----
    unsup = UnsupervisedFeatures(n_components=args.svd_components,
                                 n_day_clusters=args.day_clusters, seed=args.seed)
    unsup.fit(Xtr, pool_meta)
    Xtr_u = unsup.transform(Xtr, pool_meta)
    cats = [c for c in F.CATEGORICAL if c in Xtr_u.columns] + ["day_regime"]

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
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset,
                                        filter_target=False)
        Xva_u = D.align_columns(Xtr_u, unsup.transform(
            D.align_columns(Xtr, Xva), meta_va))

    params = dict(frozen["params"]) if frozen else dict(PARAMS)
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed, "device_type": args.device})

    print("[{}] train n={} feats={} (+{} unsupervised) cats={}".format(
        ARM, len(Xtr_u), Xtr_u.shape[1], len(unsup.added_columns), len(cats)))

    dtr = lgb.Dataset(Xtr_u, label=ytr_a, categorical_feature=cats, free_raw_data=False)
    if refit_only:
        ratio = len(Xtr_u) / max(1, frozen["n_train"])
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
        dva = lgb.Dataset(Xva_u, label=np.asarray(yva, dtype="float64"),
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
        vp = np.clip(booster.predict(Xva_u, num_iteration=booster.best_iteration),
                     config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, vp)
        print("[{}] val RMSE (clipped) = {:.3f}  (best_iter={}, early stop fired={})"
              .format(ARM, val_rmse_clipped, booster.best_iteration,
                      not conv.get("hit_max_epochs", True)))

    # how much did the unsupervised columns actually get used?
    imp = pd.DataFrame({"feature": booster.feature_name(),
                        "gain": booster.feature_importance("gain"),
                        "split": booster.feature_importance("split")})
    imp.sort_values("gain", ascending=False).to_parquet(
        out / "feature_importance.parquet", index=False)
    added = [c for c in unsup.added_columns if c in set(imp.feature)]
    share = float(imp.loc[imp.feature.isin(added), "gain"].sum()
                  / max(1e-9, imp.gain.sum()))
    ranks = {r.feature: int(i + 1) for i, r in enumerate(
        imp.sort_values("gain", ascending=False).itertuples()) if r.feature in added}
    print("  unsupervised columns take {:.1%} of total gain; ranks: {}".format(
        share, dict(sorted(ranks.items(), key=lambda kv: kv[1]))))

    sel = declare_selection_set("lightgbm_unsup best_iteration",
                                selected_on="frozen" if refit_only else "val",
                                value=int(booster.best_iteration))
    booster.save_model(str(out / "model.txt"), num_iteration=booster.best_iteration)
    with open(out / "unsup.pkl", "wb") as fh:
        pickle.dump(unsup, fh)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "unsupervised": {
            "svd_components": args.svd_components,
            "day_clusters": args.day_clusters,
            "added_columns": unsup.added_columns,
            "gain_share_of_added_columns": share,
            "gain_rank_of_added_columns": ranks,
            "svd_explained_variance_ratio": getattr(unsup, "svd_explained_", None),
            "fitted_on": "training pool only; applied unchanged to val/test/ranking",
            "uses_target": False,
        },
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration": int(booster.best_iteration),
        "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "convergence": conv, "selection": dict(sel),
        "n_train": int(len(Xtr_u)), "n_features": int(Xtr_u.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr_u.index),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
