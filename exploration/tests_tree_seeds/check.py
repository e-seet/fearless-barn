#!/usr/bin/env python3
"""Pre-registered check for tree-seed averaging (PREREGISTERED.md).

Condition 1 (decisive): seasonal_full test (Jan+Jul). 3-member blend
[FTT6, tree, MLP6] with per-airport weights solved on forward_inner (6-seed MLP,
as confirm_seasonal_full.py), tree = seed 42 vs mean of seeds 42-47.
Conditions 2-4 are read off the same construction on forward val/test (12-seed MLP).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
from common import dataset as D
from common.predict import predict_arm
from pipeline_airport_operator_catboost_multimodel.train import solve_weights_per_airport_simplex
from tests_ftt.solve_vs_strong_mlp import surface
from tests_ftt.check_6seed import blend, ftt_mean

FS, POL, SEEDS = "with_aobt", "keep_all", list(range(42, 48))
CACHE = HERE / "cache"
F = ROOT / "tests_ftt"
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
fi = lambda s: F / f"preds_seed{s}_forward_inner_lr1e-3.npz"
fw = lambda s: F / (f"preds_seed{s}_lr1e-3_e132.npz" if s <= 44 else f"preds_seed{s}_lr1e-3_e175.npz")
sf = lambda s: F / f"preds_seed{s}_seasonal_full_lr1e-3.npz"


def trees(split, part):
    f = CACHE / f"tree6_{split}_{part}.npy"
    if f.exists():
        return np.load(f)
    Xtr, _, _ = D.load_part(split, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(split, part, FS, filter_target=False)
    X = D.align_columns(Xtr, X)
    P = np.stack([predict_arm("airport_operator_blend", split, part, FS, s, POL, X=X, meta=meta, Xtr=Xtr)
                  for s in SEEDS])
    np.save(f, P)
    return P


def evaluate(mlp_seeds, surfaces):
    yw, apw, _, mw = surface("forward_inner", "val", mlp_seeds)
    Tw = trees("forward_inner", "val")
    f6 = ftt_mean(SEEDS, fi, "pv")
    W1 = solve_weights_per_airport_simplex(yw, f6, Tw[0], mw, apw, l2_shrinkage=0.0)
    W6 = solve_weights_per_airport_simplex(yw, f6, Tw.mean(0), mw, apw, l2_shrinkage=0.0)
    out = {"lirf_tree_weight_6": W6["LIRF"][1]}
    for split, part, key, fpath in surfaces:
        y, ap, _, mlp = surface(split, part, mlp_seeds)
        T = trees(split, part)
        f = ftt_mean(SEEDS, fpath, key)
        b1, b6 = blend(W1, f, T[0], mlp, ap), blend(W6, f, T.mean(0), mlp, ap)
        ib, L = y <= 7200, ap == "LIRF"
        r = {"in_band": (rm(y[ib], b1[ib]), rm(y[ib], b6[ib])), "pooled": (rm(y, b1), rm(y, b6)),
             "lirf_identical": bool(np.allclose(b1[L], b6[L]))}
        out[f"{split}_{part}"] = r
        print(f"{split:13s} {part:4s}: in-band {r['in_band'][0]:.3f} -> {r['in_band'][1]:.3f} "
              f"({r['in_band'][1]-r['in_band'][0]:+.3f}) | pooled {r['pooled'][0]:.2f} -> {r['pooled'][1]:.2f} "
              f"({r['pooled'][1]-r['pooled'][0]:+.3f}) | LIRF identical {r['lirf_identical']}")
    return out


def main() -> int:
    cfg = json.loads((F / "confirm_seasonal_full.json").read_text())
    res = {}
    res.update(evaluate(list(range(42, 54)), [("forward", "val", "pv", fw), ("forward", "test", "pt", fw)]))
    s = evaluate(cfg["mlp_seeds"], [("seasonal_full", "val", "pv", sf), ("seasonal_full", "test", "pt", sf)])
    res.update({k: v for k, v in s.items() if k.startswith("seasonal")})
    d = lambda k, m: res[k][m][1] - res[k][m][0]
    c1 = d("seasonal_full_test", "in_band") < 0
    c2 = d("forward_val", "in_band") < 0 and d("forward_test", "in_band") < 0
    c3 = d("seasonal_full_test", "pooled") <= 2.0
    c4 = all(res[k]["lirf_identical"] for k in res if isinstance(res[k], dict) and "lirf_identical" in res[k]) \
        and res["lirf_tree_weight_6"] < 1e-6
    res["verdict"] = {"jan_jul_in_band_better": c1, "forward_same_sign": c2, "jan_jul_pooled_guard": c3,
                      "lirf_unchanged": c4, "PASS": bool(c1 and c2 and c3 and c4)}
    print(f"\nPRE-REGISTERED: Jan+Jul in-band better {c1} | forward same sign {c2} | "
          f"pooled guard {c3} | LIRF unchanged {c4}  =>  {'PASS' if res['verdict']['PASS'] else 'FAIL'}")
    (HERE / "check.json").write_text(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
