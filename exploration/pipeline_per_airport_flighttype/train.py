#!/usr/bin/env python3
"""Arm per_airport_flighttype: per-(airport, flight-type bucket) LightGBM.

The hypothesis
--------------
per_airport already segments by ADEP; this arm asks a second, nested question:
within an airport, does scheduled traffic ('S', ~95.3% of rows) behave
differently enough from everything else (charter/other/NaN, bucketed as
'Other') that giving it a dedicated booster beats a single per-airport model
that has to fit both regimes with one tree structure and one stopping round?

Built to test, not to guess
----------------------------
This is the "airport x timing/flight-type" segmentation the user asked to
verify rather than have reasoned about. The cardinality worry raised earlier
was real (full airport x flight_type is up to 60 cells, some in the single
digits -- 'M' is 410 rows total, across ALL 10 airports) but S-vs-Other keeps
every one of the 20 (airport, bucket) cells above ~1,300 rows in train
(checked empirically -- see pipeline_per_airport_flighttype/model.py's
docstring). A pooled-per-airport fallback still guards any cell that comes in
below `--min-cell-rows` at fit time, so a seed with an unlucky split degrades
to per_airport for that one cell rather than training on too little data.

Gates this arm has to clear like any other
------------------------------------------
Same pool-fingerprint discipline as per_airport: the fingerprint is taken over
the union of what every airport-level fit consumed (the pooled-per-airport
booster is always fit on the airport's full row set, so no row is dropped by
routing regardless of which bucket cells got a specific booster).
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
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_per_airport_flighttype.model import (PerAirportFlightTypeLGBM,
                                                    DEFAULT_MIN_CELL_ROWS)

ARM = "per_airport_flighttype"

# Deliberately the same baseline params as per_airport -- this arm answers
# "does a second segmentation axis on top of airport help?", not "does a
# different hyperparameter config help?".
PARAMS = {
    "objective": "l2", "metric": "rmse",
    "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 100,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "max_cat_threshold": 64, "cat_smooth": 20.0,
    "verbosity": -1, "num_threads": 0,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--num-boost-round", type=int, default=5000,
                    help="SAFETY CAP per cell, not the stopping condition")
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--min-cell-rows", type=int, default=DEFAULT_MIN_CELL_ROWS)
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
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

    if not config.SPLITS[args.split]["val"]:
        raise SystemExit("split '{}' has no validation set; this arm has no "
                          "refit-only path yet.".format(args.split))
    Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset,
                                    filter_target=False)
    Xva = D.align_columns(Xtr, Xva)

    params = dict(PARAMS)
    params.update({"seed": args.seed, "bagging_seed": args.seed,
                   "feature_fraction_seed": args.seed, "device_type": args.device})

    n_ap = pool_meta[AIRPORT_COL].nunique()
    print("[{}] train n={} val n={} feats={} cats={} airports={} min_cell_rows={}".format(
        ARM, len(Xtr), len(Xva), Xtr.shape[1], len(cats), n_ap, args.min_cell_rows))

    model = PerAirportFlightTypeLGBM(params, args.num_boost_round, args.patience,
                                     cats, min_cell_rows=args.min_cell_rows)
    model.fit(Xtr, ytr, pool_meta, Xva, yva, meta_va, lgb=lgb)

    # ---- pool fingerprint over the UNION of what every sub-model consumed ----
    consumed = np.concatenate([
        Xtr.index.to_numpy()[pool_meta[AIRPORT_COL].astype("object").to_numpy() == a]
        for a in sorted(pd.unique(pool_meta[AIRPORT_COL].astype("object")))])
    if len(consumed) != len(Xtr) or set(consumed) != set(Xtr.index.to_numpy()):
        raise SystemExit(
            "per-airport routing consumed {} of {} training rows -- a segment was "
            "dropped, and the pool fingerprint would misrepresent it."
            .format(len(consumed), len(Xtr)))
    pool_sha = D.pool_fingerprint(consumed)

    vp = np.clip(model.predict(Xva, meta_va), config.PREDICTION_CLIP_MIN_SEC,
                 config.PREDICTION_CLIP_MAX_SEC)
    val_rmse_clipped = rmse(yva, vp)
    print("[{}] val RMSE (clipped) = {:.3f}   fallback used for cells: {}".format(
        ARM, val_rmse_clipped, model.used_fallback_for_ or "none"))

    capped = [k for k, c in model.convergence_.items() if c["hit_max_rounds"]]
    if capped:
        print("  WARNING: cells {} hit the round cap -- raise it; those are not "
              "converged results.".format(capped))

    sel = declare_selection_set(
        "per_airport_flighttype best_iteration (one per airport/bucket cell)",
        selected_on="val",
        value={**{k: int(v) for k, v in model.best_iter_.items()},
               **{"{}|POOLED".format(k): int(v) for k, v in model.pooled_best_iter_.items()}})

    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "min_cell_rows": args.min_cell_rows,
        "best_iteration_per_cell": {k: int(v) for k, v in model.best_iter_.items()},
        "n_train_per_cell": dict(model.n_train_),
        "convergence_per_cell": model.convergence_,
        "cells_using_pooled_fallback": model.used_fallback_for_,
        "any_cell_hit_round_cap": bool(capped),
        "val_rmse_clipped": val_rmse_clipped,
        "refit_only": False,
        "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "n_airports": int(n_ap),
        "train_pool_ids_sha": pool_sha,
        "pool_fingerprint_note": ("hashed over the UNION of rows consumed across "
                                  "all per-airport fits (the pooled-per-airport "
                                  "fallback always sees the full airport, so no "
                                  "row is dropped by bucket routing)"),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
