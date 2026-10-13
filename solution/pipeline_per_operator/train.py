#!/usr/bin/env python3
"""Arm per_operator: one LightGBM per (bucketed) aircraft operator.

The hypothesis
--------------
Only airport has ever been tried as a segmentation axis in this build. Operator
is a real, plausible one too -- ground-handling contracts, turnaround procedure
and fleet mix all vary by operator -- but `AIRCRAFT_OPERATOR_flt` is anonymised
(hashed) and never gets more than a single native-categorical role in any pooled
arm. This arm asks the same question per_airport asked, of a different column:
does routing to a dedicated booster beat asking one tree to split on the column
itself.

Measured before this was written (an isolated scratch check over the same
cached features, not this code path): per-operator alone scores ~277.8 val
RMSE, matching per_airport's ~278.2 (base protocol, seed 42) -- and the two
are complementary enough that a plain average of their predictions
(~271.9) beats either alone, and beats every other tree-based arm in this
build including per_airport_tuned. See pipeline_airport_operator_blend/ for
the formal version of that blend, with weights solved honestly rather than on
val.

Bucketing
---------
See pipeline_per_operator/model.py: the vocabulary is the top `--top-n`
operators by TRAINING volume (default 20); everything else -- rare operators
and missing values alike -- shares one "Other" booster. Learned at fit time
only, so it cannot leak a val/test-informed choice of vocabulary.

Gates this arm has to clear like any other
------------------------------------------
Same pool-fingerprint discipline as per_airport: the fingerprint is hashed over
the union of rows every bucket's sub-model consumed, asserted equal to the
incoming training index before hashing, so a silently dropped bucket fails here
rather than sailing through compare.py's equality gate.
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
from pipeline_per_operator.model import PerOperatorLGBM, OPERATOR_COL, DEFAULT_TOP_N

ARM = "per_operator"

# Deliberately the POOLED arm's baseline parameters, not the tuned ones -- same
# reasoning as per_airport: the question is "does splitting by operator help?",
# not "does splitting by operator AND retuning help?".
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
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                    help="operator vocabulary size; everything else -> Other")
    ap.add_argument("--num-boost-round", type=int, default=5000,
                    help="SAFETY CAP per bucket, not the stopping condition")
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

    # OPERATOR_COL lives in the feature frame (X), not meta.
    n_raw_operators = Xtr[OPERATOR_COL].astype(str).nunique()
    print("[{}] train n={} val n={} feats={} cats={} raw operators={} top_n={}".format(
        ARM, len(Xtr), 0 if refit_only else len(Xva), Xtr.shape[1], len(cats),
        n_raw_operators, args.top_n))

    # Pass X itself where per_airport would pass meta -- unlike ADEP_mvt,
    # AIRCRAFT_OPERATOR_flt lives in the feature frame, not meta.
    model = PerOperatorLGBM(params, args.num_boost_round, args.patience, cats,
                            top_n=args.top_n)
    model.fit(Xtr, ytr, Xtr, Xva, yva, Xva, lgb=lgb, frozen=frozen)

    # ---- pool fingerprint over the UNION of what every bucket consumed ----
    op_col = Xtr[OPERATOR_COL].astype(str)
    bucketed = op_col.where(op_col.isin(model.vocab_), "Other")
    consumed = np.concatenate([
        Xtr.index.to_numpy()[bucketed.to_numpy() == g]
        for g in sorted(pd.unique(bucketed))])
    if len(consumed) != len(Xtr) or set(consumed) != set(Xtr.index.to_numpy()):
        raise SystemExit(
            "per-operator routing consumed {} of {} training rows -- a bucket "
            "was dropped, and the pool fingerprint would misrepresent it."
            .format(len(consumed), len(Xtr)))
    pool_sha = D.pool_fingerprint(consumed)

    if refit_only:
        val_rmse_clipped = None
    else:
        vp = np.clip(model.predict(Xva, Xva), config.PREDICTION_CLIP_MIN_SEC,
                     config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_clipped = rmse(yva, vp)
        print("[{}] val RMSE (clipped) = {:.3f}".format(ARM, val_rmse_clipped))

    capped = [g for g, c in model.convergence_.items() if c["hit_max_rounds"]]
    if capped and not refit_only:
        print("  WARNING: buckets {} hit the round cap -- raise it; those are not "
              "converged results.".format(capped))

    sel = declare_selection_set(
        "per_operator best_iteration (one per bucket) + vocabulary",
        selected_on="frozen" if refit_only else "val",
        value={"vocab": model.vocab_, "best_iteration": dict(model.best_iter_)})

    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "params": params, "top_n": args.top_n,
        "vocab": model.vocab_,
        "num_boost_round_cap": args.num_boost_round, "patience": args.patience,
        "best_iteration_per_operator": {k: int(v) for k, v in model.best_iter_.items()},
        "n_train_per_operator": dict(model.n_train_),
        "convergence_per_operator": model.convergence_,
        "any_bucket_hit_round_cap": bool(capped),
        "val_rmse_clipped": val_rmse_clipped,
        "refit_only": bool(refit_only),
        "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "n_raw_operators": int(n_raw_operators), "n_buckets": len(model.boosters_),
        "train_pool_ids_sha": pool_sha,
        "pool_fingerprint_note": ("hashed over the UNION of rows consumed across "
                                  "all per-operator-bucket sub-models, asserted "
                                  "equal to the incoming training index"),
        "outlier_policy": dict(pool_meta.attrs.get("outlier_policy", {})),
        "categorical_features": cats,
    }, force=args.force)
    print("[{}] saved -> {}".format(ARM, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
