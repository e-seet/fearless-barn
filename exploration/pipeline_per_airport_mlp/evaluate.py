#!/usr/bin/env python3
"""Evaluate pipeline_per_airport_mlp arm."""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import evaluate as E
from common.bootstrap import run_dir, seed_everything, write_json_guarded
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_per_airport_mlp.model import AirportTabularMLP
from pipeline_per_airport_mlp.train import TabularAirportDataset, encode_airport_data

ARM = "per_airport_mlp"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--part", default="val", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    state_dicts = torch.load(out / "models.pt", map_location="cpu")
    with open(out / "preprocessors.pkl", "rb") as fh:
        preps = pickle.load(fh)

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    X, y, meta = D.load_part(args.split, args.part, args.featureset, filter_target=False)

    airports = sorted(preps.keys())
    pred = np.zeros(len(X), dtype="float64")

    for a in airports:
        m = (meta[AIRPORT_COL] == a)
        if m.sum() == 0:
            continue
        X_ap = X[m].drop(columns=[AIRPORT_COL], errors="ignore")
        prep = preps[a]

        cat_tensors, num_tensor = encode_airport_data(
            X_ap, prep["cat_cols"], prep["num_cols"],
            prep["cat_maps"], prep["scaler"], prep["medians"]
        )
        model = AirportTabularMLP(prep["cat_cardinalities"], num_features=len(prep["num_cols"]),
                                  hidden_dims=(128, 64), prior_mean=prep["prior_mean"]).to(device)
        model.load_state_dict(state_dicts[a])
        model.eval()

        ap_preds = []
        n_ap = len(num_tensor)
        with torch.no_grad():
            for b_start in range(0, n_ap, args.batch_size):
                x_cat = cat_tensors[b_start: b_start + args.batch_size].to(device)
                x_num = num_tensor[b_start: b_start + args.batch_size].to(device)
                ap_preds.append(model(x_cat, x_num).cpu().numpy())

        pred[m.to_numpy()] = np.concatenate(ap_preds)

    pred = np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    res = E.evaluate(y, pred, meta, ytr, part=args.part, arm=ARM, split=args.split,
                     featureset=args.featureset, seed=args.seed)
    E.print_result(res)
    write_json_guarded(out / "eval_{}.json".format(args.part), res, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
