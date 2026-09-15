#!/usr/bin/env python3
"""Evaluate the pooled congestion-feature LightGBM arm on one split part.

The training pool is reloaded and re-attached only to recover the exact column
order the booster was fitted on -- `align_columns` against it is what keeps a
LightGBM prediction from being silently made against shuffled features.
"""
from __future__ import annotations

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
from pipeline_lightgbm_congest.train import ARM
from pipeline_per_airport_congest.extra import attach


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    import lightgbm as lgb

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))
    booster = lgb.Booster(model_file=str(out / "model.txt"))
    meta_json = json.loads((out / "model.json").read_text())

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    Xtr_u = attach(Xtr, "train2025", args.featureset)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)
    X_u = D.align_columns(Xtr_u, attach(X, "train2025", args.featureset))
    pred = np.clip(booster.predict(X_u), config.PREDICTION_CLIP_MIN_SEC,
                   config.PREDICTION_CLIP_MAX_SEC)
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"best_iteration": meta_json["best_iteration"],
                            "extra_features": meta_json["extra_features"],
                            "convergence": meta_json["convergence"]})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
