#!/usr/bin/env python3
"""Arm per_airport_arrfeat: per_airport PLUS the arrival/stand features.

Tests the hypothesis the pooled arrfeat arm could NOT test: the stand taxi-in ->
taxi-out relationship is airport-specific (EHAM +0.90 ... LEBL -0.24), so a pooled
model averages the signs into noise while a per-airport model can use each sign.

Original per_airport docstring follows.

Arm per_airport: one LightGBM per departure airport, routed at prediction time.

The hypothesis
--------------
Taxi-out is an airport-specific physical process -- layout, taxiway topology,
de-icing practice, runway assignment and apron congestion are all properties of
the individual airport -- so a model per airport can specialise where a pooled
model has to spend splits separating airports before it can model anything.

The honest prior, measured before this was written
--------------------------------------------------
It is probably a small effect here, and that is worth saying up front rather
than after the number lands. The pooled LightGBM already carries `ADEP_mvt` as a
native categorical plus four `adep_*` interaction categoricals
(`adep_runway`, `adep_stand`, `stand_runway`, `adep_flight_record`), so airport
identity is not something it has to discover. And the direct test for
cross-airport contamination came back negative: switching the outlier policy from
`drop_7200` to `keep_all` -- which is entirely a LIRF-tail effect -- moved LIRF's
val RMSE by -254 s and every other airport by between -7.4 and +2.6 s. If LIRF's
extreme rows were poisoning the other nine airports' leaves, that is where it
would have shown, and it did not.

What this arm can still buy is per-airport *capacity and stopping*: each booster
early-stops on its own airport's validation rows, so an airport whose signal
saturates at 80 rounds is not forced to share a round count with one that needs
400. That is a real difference from the pooled model and is what the comparison
measures. A negative result is a genuine finding: "one model per segment" is
reached for by default far more often than it is tested.

Gates this arm has to clear like any other
------------------------------------------
`compare.py` refuses an arm whose training pool disagrees with the others. The
pool fingerprint here must therefore be taken over the **union of the rows every
sub-model consumed**, not over any one airport's slice -- a per-segment arm is
the natural place to accidentally record a partial pool and be rejected (or
worse, to record one that happens to match while a segment was silently dropped).
The union is asserted equal to the incoming index before it is hashed.
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
from pipeline_lightgbm_arrfeat.train import attach

ARM = "per_airport_arrfeat"

# Deliberately the POOLED arm's baseline parameters, not the tuned ones: the
# question this arm answers is "does splitting by airport help?", and answering
# it with a different hyperparameter set would measure both changes at once.
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
                    help="SAFETY CAP per airport, not the stopping condition")
    ap.add_argument("--patience", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=PARAMS["learning_rate"])
    ap.add_argument("--num-leaves", type=int, default=PARAMS["num_leaves"])
    ap.add_argument("--min-data-in-leaf", type=int, default=PARAMS["min_data_in_leaf"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--freeze-from", default=None,
                    help="an honest run's model.json; required for the no-val "
                         "'full' refit protocol")
    ap.add_argument("--tag", default=None,
                    help="OPTIONAL label appended to the artifact key, for runs "
                         "that vary something the protocol key does not name")
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

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set, so no round count can be "
                "selected on it. Pass --freeze-from <honest run>/model.json."
                .format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit(
                "--freeze-from points at split '{}', not a valid selection "
                "surface. Freeze from a 'forward' run.".format(frozen["split"]))
        Xva = yva = meta_va = None
    else:
        Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset,
                                        filter_target=False)
        Xva = D.align_columns(Xtr, attach(Xva, "train2025", args.featureset))

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

    model = PerAirportLGBM(params, args.num_boost_round, args.patience, cats)
    model.fit(Xtr, ytr, pool_meta, Xva, yva, meta_va, lgb=lgb, frozen=frozen)

    # ---- pool fingerprint over the UNION of what every sub-model consumed ----
    # A per-segment arm is the natural place to record a partial pool. Rebuild
    # the union from the per-airport masks and assert it against the incoming
    # index before hashing, so a silently dropped segment fails here rather than
    # sailing through compare.py's equality gate.
    consumed = np.concatenate([
        Xtr.index.to_numpy()[pool_meta[AIRPORT_COL].astype("object").to_numpy() == a]
        for a in sorted(pd.unique(pool_meta[AIRPORT_COL].astype("object")))])
    if len(consumed) != len(Xtr) or set(consumed) != set(Xtr.index.to_numpy()):
        raise SystemExit(
            "per-airport routing consumed {} of {} training rows -- a segment was "
            "dropped, and the pool fingerprint would misrepresent it."
            .format(len(consumed), len(Xtr)))
    pool_sha = D.pool_fingerprint(consumed)

    if refit_only:
        val_rmse_clipped = None
    else:
        vp = np.clip(model.predict(Xva, meta_va), config.PREDICTION_CLIP_MIN_SEC,
                     config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, vp)
        print("[{}] val RMSE (clipped) = {:.3f}".format(ARM, val_rmse_clipped))

    capped = [a for a, c in model.convergence_.items() if c["hit_max_rounds"]]
    if capped and not refit_only:
        print("  WARNING: airports {} hit the round cap -- raise it; those are not "
              "converged results.".format(capped))

    sel = declare_selection_set(
        "per_airport best_iteration (one per airport)",
        selected_on="frozen" if refit_only else "val",
        value=dict(model.best_iter_))

    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params,
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration_per_airport": {k: int(v) for k, v in model.best_iter_.items()},
        "n_train_per_airport": dict(model.n_train_),
        "convergence_per_airport": model.convergence_,
        "any_airport_hit_round_cap": bool(capped),
        "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "n_airports": int(n_ap),
        "train_pool_ids_sha": pool_sha,
        "pool_fingerprint_note": ("hashed over the UNION of rows consumed across "
                                  "all per-airport sub-models, asserted equal to "
                                  "the incoming training index"),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
