#!/usr/bin/env python3
"""Arm 1 (classical floor): no learning at all -- a proxy read plus a lookup table.

Two things this arm is:
  * the analogue of a shipped operational forecast (cf. subfolder63, where the
    grid operator's own day-ahead forecast was the real baseline to beat). With
    `--featureset with_aobt` the prediction is simply takeoff minus actual
    off-block, i.e. the taxi-out the organisers could have computed themselves.
  * a genuine hierarchical-median floor when that proxy is unavailable
    (`--featureset no_aobt`, or a row with no flight record), backing off
    stand+runway -> stand -> runway -> airport -> global.

"Fitting" here is counting medians on the training pool. It is still fitted on
TRAIN only and applied unchanged to val/test/ranking, because a median taken
over the evaluation set is an oracle, not a baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common.bootstrap import run_dir, seed_everything, write_json_guarded

ARM = "naive_offblock"
BACKOFF = [
    ("stand_runway", ["stand_runway"]),
    ("adep_stand", ["adep_stand"]),
    ("adep_runway", ["adep_runway"]),
    ("adep", ["ADEP_mvt"]),
]


def fit(X: "pd.DataFrame", y: "pd.Series") -> dict:
    tables = {}
    for name, keys in BACKOFF:
        g = y.groupby([X[k].astype("object") for k in keys], observed=True).agg(["median", "size"])
        # a level seen fewer than 20 times is noise, not a median -- back off instead
        g = g[g["size"] >= 20]
        tables[name] = g["median"]
    return {"tables": tables, "global_median": float(y.median()),
            "n_train": int(len(y))}


def predict(model: dict, X: "pd.DataFrame", featureset: str) -> "np.ndarray":
    n = len(X)
    pred = np.full(n, np.nan, dtype="float64")
    source = np.array(["global"] * n, dtype=object)

    if featureset == "with_aobt" and "delta_aobt_sec" in X.columns:
        p = X["delta_aobt_sec"].to_numpy(dtype="float64")
        ok = np.isfinite(p)
        pred[ok] = p[ok]
        source[ok] = "aobt_proxy"

    for name, keys in BACKOFF:
        need = ~np.isfinite(pred)
        if not need.any():
            break
        tbl = model["tables"][name]
        key = X[keys[0]].astype("object") if len(keys) == 1 else None
        vals = pd.Series(key).map(tbl).to_numpy(dtype="float64")
        take = need & np.isfinite(vals)
        pred[take] = vals[take]
        source[take] = name

    pred[~np.isfinite(pred)] = model["global_median"]
    # the floor arm gets the same sanity clip a learned arm's training pool has
    return np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC), source


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES),
                    help="how the TRAINING tail is handled; part of the artifact "
                         "key. Evaluation sets are never filtered.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                 filter_target=True,
                                 outlier_policy=args.outlier_policy)
    model = fit(Xtr, ytr)

    payload = {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "n_train": model["n_train"],
        "global_median": model["global_median"],
        "table_sizes": {k: int(len(v)) for k, v in model["tables"].items()},
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(_pool_meta.attrs.get("outlier_policy", {})),
        "refit_only": not config.SPLITS[args.split]["val"],
        "note": "no learned parameters; medians counted on the train pool only. "
                "Has no hyperparameters, so the 'full' refit protocol needs no "
                "--freeze-from -- there is nothing to freeze.",
    }
    for name, tbl in model["tables"].items():
        tbl.rename("median").to_frame().to_parquet(out / "table_{}.parquet".format(name))
    write_json_guarded(out / "model.json", payload, force=args.force)
    print("[{}] fitted on n={} -> {}".format(ARM, model["n_train"], out))
    for k, v in payload["table_sizes"].items():
        print("   table {:14s} levels={}".format(k, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
