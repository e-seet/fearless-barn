#!/usr/bin/env python3
"""Predictions of an airport_operator_blend run on the held-out parts used by PREREGISTERED_tree.md, via one
code path for both the old (untagged) and the adsb runs. Writes tree_preds/<name>_<split>_<part>.npy.
usage: dump_tree_preds.py old|adsb"""
import pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from pipeline_airport_operator_blend.evaluate import blend
from attach import attach

name = sys.argv[1]
OUT = HERE / "tree_preds"; OUT.mkdir(exist_ok=True)
for split, parts in (("seasonal_full", ["test"]), ("forward_inner", ["val"]), ("forward", ["val", "test"])):
    rd = config.ARTIFACTS / "runs" / ("airport_operator_blend__{}-with_aobt-keep_all__seed42{}".format(
        split, "" if name == "old" else "__tag=adsb"))
    bundle = pickle.load(open(rd / "model.pkl", "rb"))
    assert bool(bundle.get("adsb", False)) == (name == "adsb"), rd
    Xtr, _, _ = D.load_part(split, "train", "with_aobt", filter_target=True, outlier_policy="keep_all")
    if bundle.get("adsb"):
        Xtr = attach(Xtr)
    for part in parts:
        X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
        if bundle.get("adsb"):
            X = attach(X)
        X = D.align_columns(Xtr, X)
        p = np.clip(blend(bundle, X, meta), config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        np.save(OUT / f"{name}_{split}_{part}.npy", p)
        ok = np.isfinite(np.asarray(y, float))
        print(f"{name} {split} {part}: n={len(p)} tree RMSE {np.sqrt(np.mean((np.asarray(y, float)[ok] - p[ok]) ** 2)):.2f}", flush=True)
