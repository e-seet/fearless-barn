#!/usr/bin/env python3
"""Prediction-floor check (PREREGISTERED.md). No training."""
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
from pipeline_airport_operator_catboost_multimodel.train import solve_weights_per_airport_simplex
from tests_ftt.solve_vs_strong_mlp import surface
from tests_ftt.check_6seed import blend, ftt_mean
F, T = ROOT / "tests_ftt", ROOT / "tests_tabnet"
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
S6, TAB, TW, FLOORS = range(42, 48), (42, 43, 44), 0.459, (60, 120, 180, 240)
fi = lambda s: F / f"preds_seed{s}_forward_inner_lr1e-3.npz"
fw = lambda s: F / (f"preds_seed{s}_lr1e-3_e132.npz" if s <= 44 else f"preds_seed{s}_lr1e-3_e175.npz")
sf = lambda s: F / f"preds_seed{s}_seasonal_full_lr1e-3.npz"
cfg = json.loads((F / "confirm_seasonal_full.json").read_text())
out = {}
for mlp_seeds, targets in ((list(range(42, 54)), [("forward_val", "forward", "val", fw, "pv", "lr2e-3p"),
                                                  ("forward_test", "forward", "test", fw, "pt", "lr2e-3p")]),
                           (cfg["mlp_seeds"], [("seasonal_test", "seasonal_full", "test", sf, "pt", "sf_lr2e-3p")])):
    yw, apw, tw, mw = surface("forward_inner", "val", mlp_seeds)
    W = solve_weights_per_airport_simplex(yw, ftt_mean(S6, fi, "pv"), tw, mw, apw, l2_shrinkage=0.0)
    for name, split, part, fpath, key, ttag in targets:
        y, ap, tree, mlp = surface(split, part, mlp_seeds)
        p = blend(W, ftt_mean(S6, fpath, key), tree, mlp, ap)
        tab = np.mean([np.load(T / f"preds_seed{s}_{ttag}.npz")[key] for s in TAB], axis=0)
        L = ap == "LIRF"; p[L] = (1 - TW) * p[L] + TW * tab[L]
        ib = y <= 7200
        r = {}
        for fl in FLOORS:
            q = np.maximum(p, fl)
            r[fl] = {"in_band": rm(y[ib], q[ib]), "pooled": rm(y, q), "rows_raised": int((p < fl).sum())}
        out[name] = r
        print(f"{name:14s} " + " | ".join(f"floor {fl}: in-band {r[fl]['in_band']:.4f} pooled {r[fl]['pooled']:.4f} (raised {r[fl]['rows_raised']})" for fl in FLOORS))
        print(f"   y < 240 s rows on this surface: {int((y < 240).sum())} of {len(y)}")
ok = {fl: all(out[k][fl]["in_band"] < out[k][60]["in_band"] and out[k][fl]["pooled"] < out[k][60]["pooled"] for k in out) for fl in (180, 240)}
print("\nPRE-REGISTERED: better on all surfaces, both metrics ->", ok, "=>", "ADOPT " + str(max([f for f in ok if ok[f]])) if any(ok.values()) else "FAIL (keep 60)")
(HERE / "result.json").write_text(json.dumps({"surfaces": out, "verdict": ok}, indent=2, default=float))
