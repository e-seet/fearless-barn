#!/usr/bin/env python3
"""Evaluate the mlp_antigravity arm on one split part."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_mlp_antigravity.model import TabularMLP
from pipeline_mlp_antigravity.train import ARM, TabularDataset, encode_data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    meta_json = json.loads((out / "model.json").read_text())
    with open(out / "preprocessor.pkl", "rb") as fh:
        prep = pickle.load(fh)

    cat_cols = prep["cat_cols"]
    num_cols = prep["num_cols"]
    cat_maps = prep["cat_maps"]
    cat_cardinalities = prep["cat_cardinalities"]
    scaler = prep["scaler"]
    medians = prep["medians"]

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset, filter_target=True,
                              outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)

    cat_tensors, num_tensor = encode_data(X, cat_cols, num_cols, cat_maps, scaler, medians)
    ds = TabularDataset(cat_tensors, num_tensor)
    loader = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=0)

    prior_mean = prep.get("prior_mean", 990.0)
    model = TabularMLP(cat_cardinalities, num_features=len(num_cols), prior_mean=prior_mean).to(device)
    model.load_state_dict(torch.load(out / "model.pt", map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for x_cat, x_num in loader:
            x_cat = {c: x_cat[c].to(device) for c in x_cat}
            x_num = x_num.to(device)
            p = model(x_cat, x_num).cpu().numpy()
            preds.append(p)

    preds = np.concatenate(preds)
    preds = np.clip(preds, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)

    res = E.evaluate(y, preds, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed,
                     extra={"best_epoch": meta_json.get("best_epoch"),
                            "convergence": meta_json.get("convergence")})
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
