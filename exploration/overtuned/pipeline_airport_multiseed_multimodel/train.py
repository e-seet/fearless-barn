#!/usr/bin/env python3
"""Arm airport_multiseed_multimodel: Multi-seed variance reduction ensemble over airport_unsup_multimodel.

Why Multi-Seed Ensembling:
--------------------------
Neural networks (stochastic initialization and gradient steps) and gradient boosted
trees (feature/row subsampling) possess uncorrelated estimation variance across random seeds.
By combining predictions across seeds (42, 43, 44), independent variance cancels out,
achieving a mathematically guaranteed reduction in Mean Squared Error with zero leakage.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[3]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
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
from common.metrics import rmse
from common.predict import predict_arm
from common.bootstrap import (run_dir, seed_everything, write_json_guarded)

ARM = "airport_multiseed_multimodel"
BASE_ARM = "airport_unsup_multimodel"
DEFAULT_ENSEMBLE_SEEDS = [42, 43, 44]


def get_ensemble_seeds(base_seed: int) -> list[int]:
    if base_seed in (42, 43, 44):
        return [42, 43, 44]
    return [base_seed, base_seed + 1, base_seed + 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--ensemble-seeds", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    seeds = args.ensemble_seeds or get_ensemble_seeds(args.seed)
    print(f"[{ARM}] Multi-seed ensemble over {BASE_ARM} seeds {seeds}...")

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)

    preds_va = []
    for s in seeds:
        print(f"[{ARM}] Loading/generating {BASE_ARM} predictions for seed {s}...")
        p_s = predict_arm(BASE_ARM, args.split, "val", args.featureset, s,
                          args.outlier_policy, X=Xva, meta=meta_va, Xtr=Xtr)
        preds_va.append(p_s)

    pred_va = np.mean(preds_va, axis=0)
    pred_va = np.clip(pred_va, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    val_rmse = rmse(yva, pred_va)
    print(f"[{ARM}] Ensemble Val RMSE: {val_rmse:.4f}")

    model_meta = {
        "arm": ARM,
        "base_arm": BASE_ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "ensemble_seeds": seeds,
        "weights_are_val_fitted": False,
        "val_rmse": float(val_rmse),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "feature_fingerprint": config.feature_code_fingerprint(),
    }
    write_json_guarded(out / "model.json", model_meta, force=args.force)

    res_val = E.evaluate(yva, pred_va, meta_va, ytr, part="val", arm=ARM, split=args.split,
                         featureset=args.featureset, seed=args.seed,
                         extra={"ensemble_seeds": seeds, "base_arm": BASE_ARM})
    write_json_guarded(out / "eval_val.json", res_val, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
