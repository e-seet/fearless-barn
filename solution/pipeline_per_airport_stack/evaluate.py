#!/usr/bin/env python3
"""Evaluate the per-airport stacked arm on one split part."""
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
from common import evaluate as E
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_per_airport_stack.train import ARM


def blend(bundle, X, meta) -> np.ndarray:
    p_pool = bundle["pooled"].predict(
        X, num_iteration=bundle["pooled"].best_iteration)
    p_seg = bundle["seg"].predict(X, meta)
    ap = meta[AIRPORT_COL].astype("object").to_numpy()
    out = np.empty(len(X), dtype="float64")
    for a in pd.unique(ap):
        m = ap == a
        al = bundle["alphas"].get(str(a), 0.0)
        out[m] = al * p_seg[m] + (1 - al) * p_pool[m]
    return out


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
    with open(out / "model.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    meta_json = json.loads((out / "model.json").read_text())

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)
    X = D.align_columns(Xtr, X)
    pred = np.clip(blend(bundle, X, meta), config.PREDICTION_CLIP_MIN_SEC,
                   config.PREDICTION_CLIP_MAX_SEC)
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"alpha_per_airport": meta_json["alpha_per_airport"],
                            "pooled_best_iteration": meta_json["pooled_best_iteration"]})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
