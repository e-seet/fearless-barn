#!/usr/bin/env python3
"""Evaluate pipeline_airport_operator_multimodel arm."""
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
from common import evaluate as E
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_airport_operator_multimodel.train import ARM, mlp_prediction


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--tag", default=None,
                    help="Which trained variant to evaluate (matches train.py --tag).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))
    meta_json = json.loads((out / "model.json").read_text())
    weights_per_airport = meta_json["weights_per_airport"]
    # Read the MLP seed set back from the RUN's own record: the weights in this
    # file were solved against exactly that average, so evaluating with a
    # different one silently scores a model that was never fitted.
    mlp_seeds = meta_json.get("mlp_seeds") or [args.seed]

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)

    p_ap = predict_arm("airport_operator_blend", args.split, args.part,
                       args.featureset, args.seed, args.outlier_policy,
                       X=X, meta=meta, Xtr=Xtr)
    p_mlp = mlp_prediction(args.split, args.part, args.featureset, mlp_seeds,
                           args.outlier_policy, X, meta, Xtr)

    ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
    pred = np.zeros(len(y), dtype="float64")
    for a in pd.unique(ap_arr):
        m = (ap_arr == a)
        w = weights_per_airport.get(str(a), 0.5)
        pred[m] = w * p_ap[m] + (1.0 - w) * p_mlp[m]

    pred = np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"weights_per_airport": weights_per_airport,
                            "weight_split": meta_json.get("weight_split")})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
