#!/usr/bin/env python3
"""Evaluate pipeline_airport_unsup_multimodel arm."""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[3]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_airport_unsup_multimodel.train import (
    ARM, assign_clusters, _bucket_key, blend_with_cell_alpha
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)
    meta_json = json.loads((out / "model.json").read_text())
    cell_alpha = meta_json["cell_alpha"]
    airport_alpha = meta_json["airport_alpha_fallback"]
    parent_alpha = meta_json.get("parent_alpha", {})

    with open(out / "cluster_models.pkl", "rb") as fh:
        models = pickle.load(fh)
    scalers = models["scalers"]
    kmeans_models = models["kmeans_models"]

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)

    p_ap = predict_arm("airport_operator_blend", args.split, args.part,
                       args.featureset, args.seed, args.outlier_policy,
                       X=X, meta=meta, Xtr=Xtr)
    p_mlp = predict_arm("mlp_antigravity", args.split, args.part,
                        args.featureset, args.seed, args.outlier_policy,
                        X=X, meta=meta, Xtr=Xtr)

    ap_arr = meta[AIRPORT_COL].astype("object").to_numpy()
    clusters = assign_clusters(X, ap_arr, scalers, kmeans_models)
    cell_arr = _bucket_key(ap_arr, X, clusters)

    pred = blend_with_cell_alpha(p_ap, p_mlp, cell_arr, ap_arr, cell_alpha, airport_alpha, parent_alpha)
    pred = np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"cell_alpha": cell_alpha, "weight_split": meta_json.get("weight_split")})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
