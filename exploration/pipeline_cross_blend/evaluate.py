#!/usr/bin/env python3
"""Evaluate the cross-family blend arm on one split part."""
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
from common.predict import predict_arm
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_cross_blend.train import ARM, apply_weights


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

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))
    m = json.loads((out / "model.json").read_text())
    b = m["base_arms"]
    weights = m["weights"]
    if m["weight_mode"] == "global":
        weights = float(weights)

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)
    pa = predict_arm(b["a"], args.split, args.part, args.featureset, args.seed,
                     args.outlier_policy, X=X, meta=meta, Xtr=Xtr, tag=b["a_tag"])
    pb = predict_arm(b["b"], args.split, args.part, args.featureset, args.seed,
                     args.outlier_policy, X=X, meta=meta, Xtr=Xtr, tag=b["b_tag"])
    apc = meta[AIRPORT_COL].astype("object").to_numpy()
    pred = np.clip(apply_weights(pa, pb, weights, apc),
                   config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"weight_mode": m["weight_mode"], "weights": m["weights"],
                            "base_arms": b, "weight_note": m["weight_note"]})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
