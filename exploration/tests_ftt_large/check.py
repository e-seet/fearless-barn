#!/usr/bin/env python3
"""Pre-registered check for the d_token-192 FTT (PREREGISTERED.md).

FTT slot of the 3-member per-airport blend [FTT, tree, MLP]:
  comparator  small FTT, seeds 42-47 (as v32/v35)
  C1          d192 FTT, seeds 42-44
  C2          all 9 models (6 small + 3 d192), equal weight per model
Weights re-solved on forward_inner per candidate (MLP partner as check_6seed.py),
frozen, applied to forward val/test and seasonal_full test (+ Dec val, reported).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from pipeline_airport_operator_catboost_multimodel.train import solve_weights_per_airport_simplex
from tests_ftt.solve_vs_strong_mlp import surface
from tests_ftt.check_6seed import blend, ftt_mean

F = ROOT / "tests_ftt"
SMALL, LARGE = list(range(42, 48)), [42, 43, 44]
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
P = {  # (small path fn, large path fn) per surface
    "fi": (lambda s: F / f"preds_seed{s}_forward_inner_lr1e-3.npz", lambda s: F / f"preds_seed{s}_forward_inner_d192.npz"),
    "fw": (lambda s: F / (f"preds_seed{s}_lr1e-3_e132.npz" if s <= 44 else f"preds_seed{s}_lr1e-3_e175.npz"),
           lambda s: F / f"preds_seed{s}_d192.npz"),
    "sf": (lambda s: F / f"preds_seed{s}_seasonal_full_lr1e-3.npz", lambda s: F / f"preds_seed{s}_seasonal_full_d192.npz"),
}


def slots(surf, key):
    small = ftt_mean(SMALL, P[surf][0], key)
    large = ftt_mean(LARGE, P[surf][1], key)
    return {"comparator": small, "C1": large, "C2": (len(SMALL) * small + len(LARGE) * large) / (len(SMALL) + len(LARGE))}


def run(mlp_seeds, targets):
    yw, apw, tw, mw = surface("forward_inner", "val", mlp_seeds)
    W = {c: solve_weights_per_airport_simplex(yw, f, tw, mw, apw, l2_shrinkage=0.0)
         for c, f in slots("fi", "pv").items()}
    out = {"lirf_ftt_weight": {c: W[c]["LIRF"][0] for c in W},
           "ftt_weight_mean_nonlirf": {c: float(np.mean([W[c][k][0] for k in W[c] if k != "LIRF"])) for c in W}}
    for name, split, part, surf, key in targets:
        y, ap, tree, mlp = surface(split, part, mlp_seeds)
        ib = y <= 7200
        r = {}
        for c, f in slots(surf, key).items():
            b = blend(W[c], f, tree, mlp, ap)
            r[c] = {"in_band": rm(y[ib], b[ib]), "pooled": rm(y, b), "ftt_alone_in_band": rm(y[ib], f[ib])}
        out[name] = r
        print(f"{name:15s} " + " | ".join(f"{c}: in-band {r[c]['in_band']:.3f} pooled {r[c]['pooled']:.2f} "
                                          f"(FTT alone {r[c]['ftt_alone_in_band']:.1f})" for c in r))
    return out


def main() -> int:
    cfg = json.loads((F / "confirm_seasonal_full.json").read_text())
    res = run(list(range(42, 54)), [("forward_val", "forward", "val", "fw", "pv"),
                                    ("forward_test", "forward", "test", "fw", "pt")])
    s = run(cfg["mlp_seeds"], [("seasonal_val", "seasonal_full", "val", "sf", "pv"),
                               ("seasonal_test", "seasonal_full", "test", "sf", "pt")])
    res.update({k: v for k, v in s.items() if k.startswith("seasonal")})
    print("FTT weight at LIRF:", res["lirf_ftt_weight"], "| mean FTT weight off LIRF:", res["ftt_weight_mean_nonlirf"])
    verdict = {}
    for c in ("C1", "C2"):
        d = {k: res[k][c]["in_band"] - res[k]["comparator"]["in_band"] for k in ("forward_val", "forward_test", "seasonal_test")}
        c1 = all(v < 0 for v in d.values())
        c2 = res["seasonal_test"][c]["pooled"] - res["seasonal_test"]["comparator"]["pooled"] <= 2.0
        c3 = res["lirf_ftt_weight"][c] < 1e-6
        verdict[c] = {"in_band_deltas": d, "in_band_all_better": c1, "pooled_guard": c2, "lirf_zero": c3,
                      "PASS": bool(c1 and c2 and c3)}
        print(f"{c}: in-band deltas {({k: round(v, 3) for k, v in d.items()})} | pooled guard {c2} | LIRF 0 {c3} "
              f"=> {'PASS' if verdict[c]['PASS'] else 'FAIL'}")
    pick = None
    if verdict["C1"]["PASS"] and verdict["C2"]["PASS"]:
        better = all(res[k]["C2"]["in_band"] < res[k]["C1"]["in_band"] for k in ("forward_val", "forward_test", "seasonal_test"))
        pick = "C2" if better else "C1"
    elif verdict["C1"]["PASS"] or verdict["C2"]["PASS"]:
        pick = "C1" if verdict["C1"]["PASS"] else "C2"
    print(f"\nPRE-REGISTERED PICK: {pick or 'none (FAIL)'}")
    res["verdict"], res["pick"] = verdict, pick
    (HERE / "check.json").write_text(json.dumps(res, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
