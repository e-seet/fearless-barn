#!/usr/bin/env python3
"""v58 per-airport blend weights on forward_inner (tests_ftt_wide/PROBE.md + Amendment 1).

- generalised simplex solver (same SLSQP, same per-airport mean(y^2) scaling, default ftol) for N members;
  CONTROL: on v42's own 3 members it must reproduce tests_ftt/final_weights_c3d6.json exactly
- S3: [FTT slot (small, large, deep, wide at 1/4 each), tree, MLP18]
- S4: S3 + CatBoost (cb1 seeds 42-44 mean) as a fourth member
Writes final_weights_v58_s3.json and final_weights_v58_s4.json (same format as tests_ftt/final_weights_*.json,
with an extra "members" list for S4). No training."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.predict import predict_arm
from tests_tail_diagnosis.diagnose import CACHE

FS, POL, WS = "with_aobt", "keep_all", "forward_inner"
FTT = ROOT / "tests_ftt"
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def solve(y, members, airport):
    n = len(members); w0 = np.full(n, 1.0 / n)
    out = {}
    for a in sorted(pd.unique(airport)):
        m = airport == a
        P = np.column_stack([p[m] for p in members]); yv = y[m]
        scale = float(np.mean(yv ** 2)); scale = scale if scale > 0 and np.isfinite(scale) else 1.0
        obj = lambda w, P=P, yv=yv, scale=scale: float(np.mean((P @ w - yv) ** 2)) / scale
        r = minimize(obj, w0, method="SLSQP", bounds=[(0.0, 1.0)] * n,
                     constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}], options={"maxiter": 300})
        out[str(a)] = [float(x) for x in (r.x if r.success else w0)]
    return out


def ftt_slot(members, weights, key="pv", split="forward_inner"):
    return np.average([np.load(FTT / f"preds_seed{s}_{split}_{t}.npz", allow_pickle=True)[key] for s, t in members], axis=0, weights=weights)


def main():
    Xtr, _, _ = D.load_part(WS, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(WS, "val", FS, filter_target=False)
    y = np.asarray(y, float); apx = meta["ADEP_mvt"].astype(str).to_numpy()
    tree = np.load(CACHE / "tree_forward_inner_val.npy"); mlp = np.load(CACHE / "mlp18_forward_inner_val.npy")

    # control: v42's own weights
    c3 = json.loads((FTT / "final_weights_c3d6.json").read_text())
    f42 = ftt_slot([tuple(m) for m in c3["ftt_members"]], c3["ftt_member_weights"])
    W42 = solve(y, [f42, tree, mlp], apx)
    dmax = max(abs(a - b) for k in W42 for a, b in zip(W42[k], c3["weights_per_airport"][k]))
    print(f"CONTROL: generalised solver vs final_weights_c3d6.json, max |weight diff| = {dmax:.2e}")
    assert dmax < 1e-9, "generalised solver does not reproduce v42's weights"

    # v58 FTT slot: four families at 1/4 each (family totals equal)
    wide = [(s, "d256b2") for s in (42, 43, 44)]
    members = [tuple(m) for m in c3["ftt_members"]] + wide
    mw = list(c3["ftt_member_weights"]) + [2.0] * len(wide)        # small 6x1, large 9x2/3, deep 6x1, wide 3x2 -> 6 each
    f58 = ftt_slot(members, mw)
    cf = HERE / "cb1_forward_inner_val.npy"
    if not cf.exists():
        np.save(cf, np.mean([predict_arm("catboost_antigravity", WS, "val", FS, s, POL, X=X, meta=meta, Xtr=Xtr, tag="cb1") for s in (42, 43, 44)], axis=0))
    cat = np.load(cf)
    print(f"forward_inner val n={len(y)} | tree {rm(y, tree):.2f} | mlp18 {rm(y, mlp):.2f} | FTT slot v42 {rm(y, f42):.2f} -> v58 {rm(y, f58):.2f} | "
          f"wide alone {rm(y, ftt_slot(wide, None)):.2f} | CatBoost {rm(y, cat):.2f}")
    S3 = solve(y, [f58, tree, mlp], apx)
    S4 = solve(y, [f58, tree, mlp, cat], apx)
    print("\nper-airport weights  S3 [FTT, tree, MLP]  |  S4 [FTT, tree, MLP, CatBoost]")
    for k in sorted(S3):
        print(f"  {k}: " + " ".join(f"{x:.3f}" for x in S3[k]) + "  |  " + " ".join(f"{x:.3f}" for x in S4[k]))
    cat_used = [k for k in S4 if S4[k][3] > 1e-3]
    print(f"\nCatBoost weight > 0.001 at: {cat_used if cat_used else 'NO airport -> left out (Amendment 1)'}")
    print(f"LIRF FTT weight: S3 {S3['LIRF'][0]:.4f}, S4 {S4['LIRF'][0]:.4f} (must be 0)")
    common = {"mlp_seeds": list(range(42, 60)), "ftt_seeds": [s for s, _ in members], "ftt_members": [list(m) for m in members],
              "ftt_member_weights": mw, "weight_split": WS, "featureset": FS, "outlier_policy": POL,
              "solver": "per-airport simplex, default SLSQP ftol (tests_ftt_wide/solve_v58.py; reproduces c3d6 exactly)",
              "feature_fingerprint": config.feature_code_fingerprint()}
    (HERE / "final_weights_v58_s3.json").write_text(json.dumps({"weights_per_airport": S3, **common}, indent=2))
    (HERE / "final_weights_v58_s4.json").write_text(json.dumps({"weights_per_airport": S4, "members": ["ftt", "tree", "mlp", "catboost_cb1_42-44"], **common}, indent=2))
    print("wrote final_weights_v58_s3.json, final_weights_v58_s4.json")


if __name__ == "__main__":
    main()
