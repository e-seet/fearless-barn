#!/usr/bin/env python3
"""Training script for pipeline_per_airport_antigravity."""
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
from pipeline_per_airport_antigravity.model import PerAirportLGBMAntigravity, AIRPORT_COL

ARM = "per_airport_antigravity"

PARAMS = {
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--num-leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--min-data-in-leaf", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--freeze-from", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    Xtr, ytr, pool_meta = D.load_part(args.split, "train", args.featureset,
                                      filter_target=True,
                                      outlier_policy=args.outlier_policy)
    cats = [c for c in F.CATEGORICAL if c in Xtr.columns]

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit("split '{}' has no validation set. Pass --freeze-from.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        Xva = yva = meta_va = None
    else:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)

    params = dict(PARAMS)
    if frozen:
        params = dict(frozen["params"])
    else:
        params.update({"learning_rate": args.learning_rate,
                       "num_leaves": args.num_leaves,
                       "min_data_in_leaf": args.min_data_in_leaf})
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed, "device_type": args.device})

    n_ap = pool_meta[AIRPORT_COL].nunique()
    print("[{}] train n={} val n={} feats={} cats={} airports={}".format(
        ARM, len(Xtr), 0 if refit_only else len(Xva), Xtr.shape[1], len(cats), n_ap))

    model = PerAirportLGBMAntigravity(params, args.num_boost_round, args.patience, cats)
    model.fit(Xtr, ytr, pool_meta, Xva, yva, meta_va, lgb=lgb, frozen=frozen)

    consumed = np.concatenate([
        Xtr.index.to_numpy()[pool_meta[AIRPORT_COL].astype("object").to_numpy() == a]
        for a in sorted(pd.unique(pool_meta[AIRPORT_COL].astype("object")))])
    pool_sha = D.pool_fingerprint(consumed)

    val_rmse_clipped = None
    if not refit_only:
        vp = np.clip(model.predict(Xva, meta_va), config.PREDICTION_CLIP_MIN_SEC,
                     config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, vp)
        print("[{}] val RMSE (clipped) = {:.3f}".format(ARM, val_rmse_clipped))

    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)

    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "best_iteration_per_airport": {k: int(v) for k, v in model.best_iter_.items()},
        "n_train_per_airport": dict(model.n_train_),
        "convergence_per_airport": model.convergence_,
        "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "train_pool_ids_sha": pool_sha,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
