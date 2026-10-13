#!/usr/bin/env python3
"""Pre-registered check for FTT 3 -> 6 seeds (PREREGISTERED_6seed.md).

Solves the 3-member per-airport weights on forward_inner (same 12-seed MLP as
solve_vs_strong_mlp.py) twice -- W3 from FTT seeds 42-44, W6 from 42-47 -- and
applies each frozen to forward val/test and, once its seeds exist,
seasonal_full test (with the 6 seasonal_full MLP seeds, as confirm_seasonal_full.py).

usage: python tests_ftt/check_6seed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pipeline_airport_operator_catboost_multimodel.train import solve_weights_per_airport_simplex
from tests_ftt.solve_vs_strong_mlp import surface

import os
_N = int(os.environ.get("FTT_CHECK_N", "6"))  # 6: 3 vs 6 seeds; 9: 6 vs 9 seeds
S3, S6 = ([42, 43, 44], [42, 43, 44, 45, 46, 47]) if _N == 6 else (list(range(42, 48)), list(range(42, 51)))
MLP12 = list(range(42, 54))
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def ftt_mean(seeds, path_fn, key):
    zs = []
    for s in seeds:
        z = np.load(path_fn(s), allow_pickle=True)
        if "hit_cap" in z and bool(z["hit_cap"]):
            raise SystemExit(f"FTT seed {s} ({path_fn(s).name}) hit its cap -- not screenable")
        zs.append(z[key])
    return np.mean(zs, axis=0)


def blend(W, f, tree, mlp, ap):
    out = np.zeros(len(f))
    for k in np.unique(ap):
        m = ap == k
        w = W.get(str(k), [1 / 3] * 3)
        out[m] = w[0] * f[m] + w[1] * tree[m] + w[2] * mlp[m]
    return out


def main() -> int:
    fi = lambda s: HERE / f"preds_seed{s}_forward_inner_lr1e-3.npz"
    fw = lambda s: HERE / (f"preds_seed{s}_lr1e-3_e132.npz" if s <= 44 else f"preds_seed{s}_lr1e-3_e175.npz")
    sf = lambda s: HERE / f"preds_seed{s}_seasonal_full_lr1e-3.npz"

    yw, apw, tw, mw = surface("forward_inner", "val", MLP12)
    W = {}
    for nm, seeds in (("W3", S3), ("W6", S6)):
        W[nm] = solve_weights_per_airport_simplex(yw, ftt_mean(seeds, fi, "pv"), tw, mw, apw, l2_shrinkage=0.0)
    print("FTT weight per airport (W3 -> W6):")
    for k in sorted(W["W6"]):
        print(f"  {k}: {W['W3'][k][0]:.3f} -> {W['W6'][k][0]:.3f}")
    lirf_zero = W["W6"]["LIRF"][0] < 1e-6
    ref = json.loads((HERE / "solve_vs_strong_mlp.json").read_text())["w3_per_airport"]
    repro = max(abs(W["W3"][k][i] - ref[k][i]) for k in ref for i in range(3))
    print(f"W3 reproduces solve_vs_strong_mlp.json to max |diff| {repro:.2e}")

    res, conds = {}, {}
    for part, key in (("val", "pv"), ("test", "pt")):
        y, ap, tree, mlp = surface("forward", part, MLP12)
        ib = y <= 7200
        r = {nm: rm(y[ib], blend(W[nm], ftt_mean(sd, fw, key), tree, mlp, ap)[ib])
             for nm, sd in (("W3", S3), ("W6", S6))}
        L = ap == "LIRF"
        r3 = blend(W["W3"], ftt_mean(S3, fw, key), tree, mlp, ap)
        r6 = blend(W["W6"], ftt_mean(S6, fw, key), tree, mlp, ap)
        print(f"forward {part}: in-band 3-seed {r['W3']:.3f} | 6-seed {r['W6']:.3f} | "
              f"delta {r['W6'] - r['W3']:+.3f} | pooled {rm(y, r3):.2f} -> {rm(y, r6):.2f} | "
              f"LIRF in-band {rm(y[ib & L], r3[ib & L]):.2f} -> {rm(y[ib & L], r6[ib & L]):.2f}")
        res[f"forward_{part}"] = r
        conds[f"forward_{part}_better"] = r["W6"] < r["W3"]

    have_sf = all(sf(s).exists() for s in S6)
    if have_sf:
        cfg = json.loads((HERE / "confirm_seasonal_full.json").read_text())
        y, ap, tree, mlp = surface("seasonal_full", "test", cfg["mlp_seeds"])
        ib = y <= 7200
        # seasonal_full weights are solved on forward_inner with the SAME MLP seeds used on seasonal_full
        yw6, apw6, tw6, mw6 = surface("forward_inner", "val", cfg["mlp_seeds"])
        r = {}
        for nm, seeds in (("W3", S3), ("W6", S6)):
            Ws = solve_weights_per_airport_simplex(yw6, ftt_mean(seeds, fi, "pv"), tw6, mw6, apw6, l2_shrinkage=0.0)
            r[nm] = rm(y[ib], blend(Ws, ftt_mean(seeds, sf, "pt"), tree, mlp, ap)[ib])
        print(f"seasonal_full test (Jan+Jul): in-band 3-seed {r['W3']:.3f} | 6-seed {r['W6']:.3f} | "
              f"delta {r['W6'] - r['W3']:+.3f}")
        res["seasonal_full_test"] = r
        conds["seasonal_full_test_better"] = r["W6"] < r["W3"]
    else:
        print("seasonal_full test: seeds 45-47 not trained yet -- condition 4 PENDING")
    conds["lirf_weight_zero"] = lirf_zero
    verdict = ("PASS" if all(conds.values()) and have_sf else
               "FAIL" if not all(conds.values()) else "PENDING (forward conditions met)")
    print(f"\nconditions: {conds}  =>  {verdict}")
    (HERE / f"check_{_N}seed.json").write_text(json.dumps(
        {"W3": W["W3"], "W6": W["W6"], "results": res, "conditions": conds, "verdict": verdict},
        indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
