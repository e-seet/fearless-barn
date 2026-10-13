#!/usr/bin/env python3
"""Final per-airport weights for the FTT candidate, solved on forward_inner.

Matches v25's own recipe as closely as possible: the shipped submission averages
**18** MLP seeds (42-59) and solves its per-airport weights on `forward_inner`.
The earlier checks used 6 and 12 seeds because that is what exists on the
`forward` split; the candidate is built on `full`, where all 18 exist, so the
weights are solved at 18 here.

  3-member: FTT + tree + MLP18, per-airport simplex (w >= 0, sum = 1)

Solved ONLY on forward_inner val (Jul-Aug), which is a holdout carved from
TRAIN. `full` is in SELECTION_FORBIDDEN_SPLITS and nothing is fitted there;
these weights are frozen and applied to the scored set.

Writes final_weights.json. No training. Prediction-heavy (~18 MLP forward
passes), so it caches.

usage: python tests_ftt/solve_final_weights.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.predict import predict_arm
from common.bootstrap import declare_selection_set
from pipeline_airport_operator_catboost_multimodel.train import (
    solve_weights_per_airport_simplex)

FS, POL, WS = "with_aobt", "keep_all", "forward_inner"
MLP_SEEDS = list(range(42, 60))          # 18, exactly v25's set
FTT_SEEDS = [int(x) for x in os.environ.get("FTT_SEEDS", "42,43,44").split(",")]
OUT_NAME = os.environ.get("FTT_WEIGHTS_OUT", "final_weights.json")  # v30 used the default
SEED = 42
CACHE = Path(os.environ.get(
    "FTT_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
    r"\2ff7d479-271f-463d-9593-1a05404d65d1\scratchpad\ftt_strong_cache"))
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def main() -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    Xtr, _, _ = D.load_part(WS, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(WS, "val", FS, filter_target=False)
    X = D.align_columns(Xtr, X)
    y = np.asarray(y, float)
    apx = meta["ADEP_mvt"].astype(str).to_numpy()

    f = CACHE / "tree_forward_inner_val.npy"
    tree = np.load(f) if f.exists() else predict_arm(
        "airport_operator_blend", WS, "val", FS, SEED, POL, X=X, meta=meta, Xtr=Xtr)
    if not f.exists():
        np.save(f, tree)

    f18 = CACHE / "mlp18_forward_inner_val.npy"
    if f18.exists():
        mlp = np.load(f18)
    else:
        mlp = np.mean([predict_arm("mlp_antigravity", WS, "val", FS, s, POL,
                                   X=X, meta=meta, Xtr=Xtr) for s in MLP_SEEDS], axis=0)
        np.save(f18, mlp)

    members = ([(int(m.split(":")[0]), m.split(":")[1]) for m in os.environ["FTT_MEMBERS"].split(",")]
               if os.environ.get("FTT_MEMBERS") else [(s, "lr1e-3") for s in FTT_SEEDS])
    mw = ([float(m.split(":")[2]) if m.count(":") == 2 else 1.0 for m in os.environ["FTT_MEMBERS"].split(",")]
          if os.environ.get("FTT_MEMBERS") else None)
    ftt = np.average([np.load(HERE / f"preds_seed{s}_forward_inner_{tg}.npz",
                              allow_pickle=True)["pv"] for s, tg in members], axis=0, weights=mw)
    assert len(ftt) == len(y) == len(tree) == len(mlp)

    print(f"forward_inner val n={len(y)}")
    print(f"  tree {rm(y,tree):.2f} | mlp18 {rm(y,mlp):.2f} | ftt-ens3 {rm(y,ftt):.2f}")

    W = solve_weights_per_airport_simplex(y, ftt, tree, mlp, apx, l2_shrinkage=0.0)
    declare_selection_set("ftt 3-member per-airport weights",
                          selected_on="forward_inner", value=W)

    print(f"\nper-airport weights [FTT, tree, MLP18]:")
    for k in sorted(W):
        w = W[k]
        print(f"  {k}: FTT {w[0]:.4f}  tree {w[1]:.4f}  MLP {w[2]:.4f}")

    lirf = W["LIRF"][0]
    print(f"\nSANITY: FTT weight at LIRF = {lirf:.4f} "
          f"(0.000 in all four previous solves)")
    if lirf > 1e-6:
        print("  *** WARNING: non-zero. Every prior solve put FTT at exactly 0 "
              "here. Distrust this solve rather than shipping it. ***")

    blend = np.zeros(len(y))
    for k in np.unique(apx):
        m = apx == k
        w = W[str(k)]
        blend[m] = w[0] * ftt[m] + w[1] * tree[m] + w[2] * mlp[m]
    print(f"\nin-sample on the solve surface (NOT an estimate of anything): "
          f"3-member {rm(y, blend):.2f}")

    out = {"weights_per_airport": W, "mlp_seeds": MLP_SEEDS, "ftt_seeds": [s for s, _ in members],
           **({"ftt_members": [[s, tg] for s, tg in members]} if os.environ.get("FTT_MEMBERS") else {}),
           **({"ftt_member_weights": mw} if mw and any(w != 1.0 for w in mw) else {}),
           "weight_split": WS, "featureset": FS, "outlier_policy": POL,
           "solver": "per-airport simplex (w>=0, sum=1), "
                     "pipeline_airport_operator_catboost_multimodel.solve_weights_per_airport_simplex",
           "feature_fingerprint": config.feature_code_fingerprint(),
           "forward_inner_rmse": {"tree": rm(y, tree), "mlp18": rm(y, mlp),
                                  "ftt_ens3": rm(y, ftt), "three_member": rm(y, blend)}}
    (HERE / OUT_NAME).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / OUT_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
