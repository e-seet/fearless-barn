#!/usr/bin/env python3
"""Arm 2 (interpretable learned floor): ridge regression.

Linear, inspectable coefficients, and no capacity to memorise 1.4M rows -- the
'can a simple model already do this?' arm. Design matrix:
  * numeric features, median-imputed (statistics fitted on TRAIN only) + scaled
  * low-cardinality categoricals one-hot encoded
  * high-cardinality categoricals out-of-fold target encoded (common/encoders.py)

Trained on squared error, which is the metric, so the ridge optimises exactly
what the leaderboard scores.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import features as F
from common.encoders import OutOfFoldTargetEncoder
from common.metrics import rmse
from common.bootstrap import (declare_selection_set, run_dir, seed_everything,
                              write_json_guarded)

# RidgePipeline lives in pipeline_ridge/model.py, NOT here: a class defined in an
# entry script is pickled as `__main__.RidgePipeline` and cannot be unpickled by
# any other process (MISTAKES.md #78). Re-exported so `from pipeline_ridge.train
# import RidgePipeline` -- which stage4_controls.py uses -- keeps working and
# resolves to the SAME class object pickle checks identity against.
from pipeline_ridge.model import (HIGH_CARD, LOW_CARD,  # noqa: E402
                                  RidgePipeline, numeric_cols)

ARM = "ridge"



def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--alphas", default="1,10,100,1000",
                    help="comma-separated ridge alphas; selected on VAL RMSE")
    ap.add_argument("--freeze-from", default=None,
                    help="path to an honest run's model.json; required for the "
                         "no-val 'full' refit protocol (alpha arrives frozen)")
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES),
                    help="how the TRAINING tail is handled; part of the artifact "
                         "key. Evaluation sets are never filtered.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                 filter_target=True,
                                 outlier_policy=args.outlier_policy)
    refit_only = not config.SPLITS[args.split]["val"]
    if refit_only:
        # design-gate finding B4 -- the submission model's real code path.
        if not args.freeze_from:
            raise SystemExit(
                "split '{}' has no validation set, so alpha cannot be selected on "
                "it. Pass --freeze-from <honest run>/model.json.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        if frozen.get("split") in config.SELECTION_FORBIDDEN_SPLITS:
            raise SystemExit("--freeze-from points at split '{}', not a valid "
                             "selection surface".format(frozen["split"]))
        alpha = float(frozen["alpha"])
        print("[{}] REFIT on '{}' with alpha={} frozen from {}".format(
            ARM, args.split, alpha, args.freeze_from))
        best = {"alpha": alpha, "val_rmse": None,
                "pipe": RidgePipeline(args.seed, alpha).fit(Xtr, ytr)}
        sweep = []
        sel = declare_selection_set("ridge alpha", selected_on="frozen", value=alpha)
    else:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
        print("[{}] train n={} val n={}".format(ARM, len(Xtr), len(Xva)))

        sweep = []
        best = None
        for a in [float(x) for x in args.alphas.split(",")]:
            pipe = RidgePipeline(args.seed, a).fit(Xtr, ytr)
            v = rmse(yva, pipe.predict(Xva))
            sweep.append({"alpha": a, "val_rmse": v})
            print("   alpha={:<8g} val RMSE={:.3f}".format(a, v))
            if best is None or v < best["val_rmse"]:
                best = {"alpha": a, "val_rmse": v, "pipe": pipe}
        sel = declare_selection_set("ridge alpha", selected_on="val", value=best["alpha"])
    # Pickle records a class by MODULE PATH. Run as `python pipeline_ridge/train.py`
    # this class's __module__ is "__main__", so the artifact only unpickles inside
    # another __main__ that happens to define RidgePipeline -- and evaluate.py, run
    # as `-m pipeline_ridge.evaluate`, is a different __main__. Pin the canonical
    # path so the model loads under any invocation style (by path, by -m, imported).
    RidgePipeline.__module__ = "pipeline_ridge.train"
    with open(out / "model.pkl", "wb") as fh:
        pickle.dump(best["pipe"], fh)
    coef = pd.Series(best["pipe"].model.coef_, index=best["pipe"].feature_names_)
    coef.reindex(coef.abs().sort_values(ascending=False).index).head(40).to_frame(
        "coef").to_parquet(out / "top_coefficients.parquet")
    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "alpha": best["alpha"], "val_rmse": best["val_rmse"],
        "sweep": sweep, "selection": dict(sel), "refit_only": bool(refit_only),
        "n_train": int(len(Xtr)), "n_features": len(best["pipe"].feature_names_),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(_pool_meta.attrs.get("outlier_policy", {})),
    }, force=args.force)
    print("[{}] alpha={} {} -> {}".format(
        ARM, best["alpha"],
        "(refit, no val by design)" if refit_only
        else "val RMSE={:.3f}".format(best["val_rmse"]), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
