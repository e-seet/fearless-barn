#!/usr/bin/env python3
"""Evaluate the ensemble_antigravity arm on one split part."""
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
from pipeline_ensemble_antigravity.train import ARM, predict_base_arm


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

    weights = meta_json["weights"]
    base_arms = meta_json["base_arms"]

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True,
                              outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)

    preds = []
    arm_weights = []
    for arm in base_arms:
        w = weights.get(arm, 0.0)
        if w <= 0:
            continue
        p = predict_base_arm(arm, args.split, args.part, args.featureset, args.seed,
                             args.outlier_policy, X, Xtr=Xtr)
        preds.append(p)
        arm_weights.append(w)

    total_w = sum(arm_weights)
    if total_w > 0:
        norm_weights = [w / total_w for w in arm_weights]
    else:
        norm_weights = [1.0 / len(arm_weights)] * len(arm_weights)

    blend_pred = np.zeros(len(y), dtype="float64")
    for w, p in zip(norm_weights, preds):
        blend_pred += w * p

    blend_pred = np.clip(blend_pred, config.PREDICTION_CLIP_MIN_SEC,
                         config.PREDICTION_CLIP_MAX_SEC)

    res = E.evaluate(y, blend_pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"weights": weights, "base_arms": base_arms})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
