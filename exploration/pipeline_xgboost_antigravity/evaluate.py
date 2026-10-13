#!/usr/bin/env python3
"""Evaluate the xgboost_antigravity arm on one split part."""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_xgboost_antigravity.train import ARM


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES),
                    help="how the TRAINING tail is handled; part of the artifact key")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import xgboost as xgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)
    booster = xgb.Booster()
    booster.load_model(str(out / "model.ubj"))
    meta_json = json.loads((out / "model.json").read_text())

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True,
                              outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)
    X = D.align_columns(Xtr, X)
    dmat = xgb.DMatrix(X, enable_categorical=True)

    best_iter = meta_json.get("best_iteration")
    iter_range = (0, best_iter + 1) if best_iter is not None else None
    pred = booster.predict(dmat, iteration_range=iter_range)
    pred = np.clip(pred, config.PREDICTION_CLIP_MIN_SEC,
                   config.PREDICTION_CLIP_MAX_SEC)

    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"best_iteration": meta_json["best_iteration"],
                            "convergence": meta_json["convergence"]})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
