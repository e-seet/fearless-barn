"""Free diagnostic: do the blend alphas depend on whether the forward_inner MLP
seed was truncated at the 25-epoch cap or allowed to early-stop?

Capped trio   {42,43,44}: hit_max_epochs=True, early stop never fired.
Converged trio{45,46,47}: max_epochs=45, early stop fired (best 23/27/25).

No training. Loads existing forward_inner runs and re-solves the same
least-squares per-airport alpha the shipped arm uses.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(r"C:\Users\Admin\Desktop\eddie\fearless-barn\exploration")
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)
import config
from common import dataset as D
from common.metrics import rmse
from common.predict import predict_arm
from pipeline_per_airport.model import AIRPORT_COL
sys.path.insert(0, str(ROOT / "pipeline_airport_operator_multimodel"))
from pipeline_airport_operator_multimodel.train import solve_alpha_per_airport

FS, OP, WS = "with_aobt", "keep_all", "forward_inner"

Xw_tr, _, _ = D.load_part(WS, "train", FS, filter_target=True, outlier_policy=OP)
Xw, yw, meta_w = D.load_part(WS, "val", FS, filter_target=False)
yw = np.asarray(yw, dtype="float64")
ap_w = meta_w[AIRPORT_COL].astype("object").to_numpy()
print("forward_inner val rows:", len(yw))

pa = predict_arm("airport_operator_blend", WS, "val", FS, 42, OP,
                 X=Xw, meta=meta_w, Xtr=Xw_tr, tag=None)
print("tree parent val RMSE:", round(rmse(yw, pa), 3))

SEEDS = list(range(42, 60))
preds = {}
for s in SEEDS:
    try:
        preds[s] = predict_arm("mlp_antigravity", WS, "val", FS, s, OP,
                               X=Xw, meta=meta_w, Xtr=Xw_tr, tag=None)
    except Exception as e:
        print("skip seed", s, type(e).__name__, e)

conv = {}
for s in preds:
    mj = json.load(open(ROOT / f"artifacts/runs/mlp_antigravity__{WS}-{FS}-{OP}__seed{s}/model.json"))
    c = mj.get("convergence", {})
    conv[s] = (c.get("hit_max_epochs"), c.get("best_epoch"), c.get("max_epochs"))

print("\nper-seed forward_inner val RMSE (capped? best/max):")
for s in sorted(preds):
    h, b, m = conv[s]
    print(f"  seed {s}: {rmse(yw, preds[s]):8.3f}   capped={str(h):5s} best={b} max={m}")

capped = [s for s in preds if conv[s][0]]
ok = [s for s in preds if not conv[s][0]]
print("\ncapped seeds:", capped)
print("converged seeds:", ok)
print(f"mean val RMSE capped={np.mean([rmse(yw,preds[s]) for s in capped]):.3f} "
      f"converged={np.mean([rmse(yw,preds[s]) for s in ok]):.3f}")

def solve(seeds):
    pb = np.mean([preds[s] for s in seeds], axis=0)
    w = solve_alpha_per_airport(yw, pa, pb, ap_w)
    blend = np.array([w[a] for a in ap_w]) * pa + (1 - np.array([w[a] for a in ap_w])) * pb
    return w, rmse(yw, pb), rmse(yw, blend)

sets = {
    "capped trio 42,43,44": [42, 43, 44],
    "converged trio 45,46,47": [45, 46, 47],
    "all 18 (shipped v25)": SEEDS,
    "all capped (15)": sorted(capped),
    "all converged (3+)": sorted(ok),
}
res = {}
for name, ss in sets.items():
    ss = [s for s in ss if s in preds]
    res[name] = solve(ss)

airports = sorted(set(ap_w))
print("\n=== tree weight alpha per airport (MLP weight = 1-alpha) ===")
hdr = f"{'airport':<8}" + "".join(f"{n[:22]:>24}" for n in sets)
print(hdr)
for a in airports:
    print(f"{a:<8}" + "".join(f"{res[n][0][a]:>24.4f}" for n in sets))
print(f"{'MLPens':<8}" + "".join(f"{res[n][1]:>24.3f}" for n in sets))
print(f"{'blendRMSE':<8}" + "".join(f"{res[n][2]:>24.3f}" for n in sets))

d = {a: res["converged trio 45,46,47"][0][a] - res["capped trio 42,43,44"][0][a] for a in airports}
print("\nalpha shift converged-trio minus capped-trio (negative = MORE weight on MLP):")
for a in airports:
    print(f"  {a}: {d[a]:+.4f}")
print("mean shift:", round(float(np.mean(list(d.values()))), 4))
