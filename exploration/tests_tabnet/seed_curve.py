#!/usr/bin/env python3
"""Has the TabNet seed ensemble plateaued? (final models, Dec-2025 val, LIRF rows)

The final models train on Jan-Nov 2025 and early-stop on Dec 2025, so Dec val is the only
surface they can be scored on. It is slightly optimistic (the early-stopping epoch was
chosen on it) but fine for a RELATIVE seed-count curve. Uses healthy seeds only: aligned
marker, Dec val <= 260, best_epoch >= 10, early-stopped below the cap.

PRE-REGISTERED BAR (before any of seeds 49-54 finished): build a new candidate and propose a
submission only if TabNet-ensemble LIRF RMSE on Dec val improves by >= 6s going from the
original 5 healthy seeds {42,43,45,46,48} to all healthy seeds. (~0.5 board-seconds after a
50/50 mix by the conversion dR = 0.078 * L_LIRF * d / R, L~600, R~294, halved for w~0.5.)
Curve = mean over random subsets of each size k (so no seed ordering can mislead).
"""
import glob
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
Z = {}
for f in sorted(glob.glob(str(HERE / "build" / "final_seed*.npz"))):
    s = int(Path(f).stem.split("seed")[1])
    z = np.load(f, allow_pickle=True)
    ok = ("labels_aligned" in z.files) and float(z["val_rmse"]) <= 260 and int(z["best_epoch"]) >= 10 and not bool(z["hit_cap"])
    print(f"seed {s}: Dec val {float(z['val_rmse']):.1f} best_epoch {int(z['best_epoch'])} ran {int(z['epochs_run'])} -> {'HEALTHY' if ok else 'excluded'}")
    if ok:
        Z[s] = z
seeds = sorted(Z)
print(f"\nhealthy seeds ({len(seeds)}): {seeds}")
ids = Z[seeds[0]]["val_ids"]
assert all(np.array_equal(Z[s]["val_ids"], ids) for s in seeds)
y = Z[seeds[0]]["yva"]
# airport for the Dec-val rows, from the training-scope features
import sys
sys.path.insert(0, str(HERE.parent))
from common import dataset as D
import config
Xall = D.build_scope("train2025", "with_aobt").set_index(config.ID)
ap = Xall.loc[ids, "ADEP_mvt"].astype(str).to_numpy()
L = ap == "LIRF"
P = np.array([Z[s]["pv"] for s in seeds])
print(f"Dec val rows {len(y)}, LIRF rows {int(L.sum())}\n")
rng = np.random.default_rng(0)
print(f"{'k':>3s} {'LIRF RMSE':>10s} {'sd':>6s} {'pooled RMSE':>12s}")
curve = {}
for k in range(1, len(seeds) + 1):
    R = []
    for _ in range(300 if k < len(seeds) else 1):
        e = P[rng.choice(len(seeds), k, replace=False)].mean(0)
        R.append((rm(y[L], e[L]), rm(y, e)))
    R = np.array(R)
    curve[k] = (R[:, 0].mean(), R[:, 1].mean())
    print(f"{k:3d} {R[:,0].mean():10.1f} {R[:,0].std():6.1f} {R[:,1].mean():12.2f}")
orig = [42, 43, 45, 46, 48]
o = P[[seeds.index(s) for s in orig if s in seeds]].mean(0)
a = P.mean(0)
d = rm(y[L], o[L]) - rm(y[L], a[L])
print(f"\noriginal 5 seeds LIRF {rm(y[L], o[L]):.1f}  ->  all {len(seeds)} healthy seeds LIRF {rm(y[L], a[L]):.1f}   improvement {d:+.1f}s  (bar: >= 6.0)")
print("VERDICT:", "BAR MET -> build candidate" if d >= 6.0 else "BAR NOT MET -> close, nothing to submit")
