#!/usr/bin/env python3
"""Evaluate pipeline_airport_multiseed_multimodel arm."""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[3]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded

ARM = "airport_multiseed_multimodel"


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
    base_arm = meta_json.get("base_arm", "airport_unsup_multimodel")
    seeds = meta_json.get("ensemble_seeds", [42, 43, 44])

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)

    preds = []
    for s in seeds:
        p_s = predict_arm(base_arm, args.split, args.part,
                          args.featureset, s, args.outlier_policy,
                          X=X, meta=meta, Xtr=Xtr)
        preds.append(p_s)

    pred = np.mean(preds, axis=0)
    pred = np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"ensemble_seeds": seeds, "base_arm": base_arm})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
