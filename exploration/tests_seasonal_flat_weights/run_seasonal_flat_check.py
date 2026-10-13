#!/usr/bin/env python3
"""Confirms TAXI_OUT_DRIVERS.md sec.2n's fitted-vs-flat weight finding
(shipped simplex weights beat flat [1/3,1/3,1/3] by ~35s on forward's test)
also holds on the board-composition-matched `seasonal` split, not just
`forward`. Zero training: re-blends predictions from arms already on disk,
exactly like sec.2o's mechanism-hunt did for the 3-way blend's overall edge.

Uses the SAME shipped per-airport weights as the real v25/v26 submission
(airport_operator_catboost_multimodel, tag=simplex_avg, forward_inner-solved)
and the same arms sec.2o already trained fresh under `seasonal`: catboost_antigravity
(seed42, tag=cb1) and airport_operator_blend (seed42) -- both single-seed
under seasonal, a known limitation already flagged in sec.2o, not new here --
plus mlp_antigravity averaged over its full 6 seasonal seeds (42-47).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_airport_operator_multimodel.train import mlp_prediction
from pipeline_airport_catboost_multimodel.train import catboost_prediction

SPLIT = "seasonal"
FEATURESET = "with_aobt"
OUTLIER_POLICY = "keep_all"
TREE_ARM = "airport_operator_blend"
MLP_SEEDS = [42, 43, 44, 45, 46, 47]
CB_SEEDS = [42]          # only seed available under seasonal (sec.2o)
TREE_SEED = 42           # only seed available under seasonal (sec.2o)
CB_TAG = "cb1"

# The exact shipped weights (airport_operator_catboost_multimodel,
# tag=simplex_avg, forward_inner-solved) used for the real v25/v26 submission.
SHIPPED_WEIGHTS = {
    "EDDF": [0.34934041615480055, 0.1703914290627293, 0.4802681547824702],
    "EDDM": [0.1690861355617381, 0.388302552485752, 0.4426113119525098],
    "EGLL": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
    "EHAM": [0.4094716233084597, 0.4724557420526511, 0.11807263463888917],
    "LEBL": [0.34658505423386815, 0.5316794063590246, 0.12173553940710713],
    "LEMD": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
    "LFPG": [0.5178013339840699, 0.14059917847320536, 0.34159948754272473],
    "LIRF": [1.0408340855860843e-17, 0.0, 1.0],
    "LSZH": [0.35181434191590866, 0.19038452757746827, 0.457801130506623],
    "LTFM": [0.25822886906549164, 0.3115212528734597, 0.4302498780610487],
}
FLAT_WEIGHTS = {a: [1 / 3, 1 / 3, 1 / 3] for a in SHIPPED_WEIGHTS}


def blend(weights, p_cb, p_tree, p_mlp, airport):
    out = np.zeros(len(p_cb), dtype="float64")
    for a in pd.unique(airport):
        w_cb, w_tree, w_mlp = weights.get(str(a), [0.0, 0.0, 1.0])
        m = (airport == a)
        out[m] = w_cb * p_cb[m] + w_tree * p_tree[m] + w_mlp * p_mlp[m]
    return np.clip(out, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def main() -> int:
    Xtr, _, _ = D.load_part(SPLIT, "train", FEATURESET, filter_target=True,
                            outlier_policy=OUTLIER_POLICY)
    Xte, yte, meta_te = D.load_part(SPLIT, "test", FEATURESET, filter_target=False)
    yte = np.asarray(yte, dtype="float64")
    airport = meta_te[AIRPORT_COL].astype("object").to_numpy()
    print("seasonal test: n={}, airports present: {}".format(
        len(yte), sorted(pd.unique(airport))))

    p_cb = catboost_prediction(SPLIT, "test", FEATURESET, CB_SEEDS, OUTLIER_POLICY,
                               Xte, meta_te, Xtr, tag=CB_TAG)
    p_tree = predict_arm(TREE_ARM, SPLIT, "test", FEATURESET, TREE_SEED, OUTLIER_POLICY,
                         X=Xte, meta=meta_te, Xtr=Xtr)
    p_mlp = mlp_prediction(SPLIT, "test", FEATURESET, MLP_SEEDS, OUTLIER_POLICY,
                           Xte, meta_te, Xtr, tag=None)

    p_shipped = blend(SHIPPED_WEIGHTS, p_cb, p_tree, p_mlp, airport)
    p_flat = blend(FLAT_WEIGHTS, p_cb, p_tree, p_mlp, airport)

    rmse_shipped = rmse(yte, p_shipped)
    rmse_flat = rmse(yte, p_flat)
    print("\npooled test RMSE:")
    print("  shipped (forward_inner-solved simplex weights): {:.3f}".format(rmse_shipped))
    print("  flat [1/3, 1/3, 1/3]:                            {:.3f}".format(rmse_flat))
    print("  flat - shipped:                                  {:+.3f}".format(rmse_flat - rmse_shipped))

    print("\nper-airport breakdown (n, shipped RMSE, flat RMSE, flat-shipped):")
    clip = lambda p: np.clip(p, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
    for a in sorted(pd.unique(airport)):
        m = (airport == a)
        rs = rmse(yte[m], p_shipped[m])
        rf = rmse(yte[m], p_flat[m])
        print("  {:6s} n={:6d}  shipped={:9.2f}  flat={:9.2f}  flat-shipped={:+8.2f}".format(
            a, int(m.sum()), rs, rf, rf - rs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
