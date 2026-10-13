#!/usr/bin/env python3
"""DESCRIPTIVE ONLY (post-hoc): does the later-epochs ensemble gain survive the
tree+MLP blend? Stage A == forward_inner (Jan-Jun -> Jul-Aug), stage B == forward.
Per-airport tree/MLP alpha solved on Jul-Aug with the MLP ensemble at uniform
epoch E (stage A), applied frozen to stage B at the same E on Sep-Oct/Nov-Dec.
Tree predictions from tests_ftt/solve_vs_strong_mlp.py's cache.
"""
import os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
from pipeline_airport_operator_multimodel.train import solve_alpha_per_airport
CACHE = Path(os.environ.get("FTT_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
    r"\2ff7d479-271f-463d-9593-1a05404d65d1\scratchpad\ftt_strong_cache"))
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
seeds = range(42, 54)
m = np.load(HERE / "meta.npz", allow_pickle=True)
cur = {s: np.load(HERE / f"curves_seed{s}.npz") for s in seeds}
tree = {"julaug": np.load(CACHE / "tree_forward_inner_val.npy"),
        "sepoct": np.load(CACHE / "tree_forward_val.npy"),
        "novdec": np.load(CACHE / "tree_forward_test.npy")}
for k in tree:
    assert len(tree[k]) == len(m[f"y_{k}"]), k
print(f"{'E':>3} | {'Sep-Oct in-band':>15} {'LIRF':>8} {'nonLIRF':>8} | {'Nov-Dec in-band':>15} {'LIRF':>8} {'nonLIRF':>8} {'pooled':>8} | mean alpha")
for E in (10, 13, 16, 20, 25, 30, 35, 40, 45):
    mlpA = np.mean([cur[s]["A_julaug"][E - 1] for s in seeds], axis=0)
    al = solve_alpha_per_airport(m["y_julaug"], tree["julaug"], mlpA, m["ap_julaug"])
    row = []
    for ev in ("sepoct", "novdec"):
        mlp = np.mean([cur[s][f"B_{ev}"][E - 1] for s in seeds], axis=0)
        y, ap = m[f"y_{ev}"], m[f"ap_{ev}"]
        a = np.array([al.get(str(x), 0.5) for x in ap])
        p = a * tree[ev] + (1 - a) * mlp
        ib, L = y <= 7200, ap == "LIRF"
        row += [rm(y[ib], p[ib]), rm(y[ib & L], p[ib & L]), rm(y[ib & ~L], p[ib & ~L])]
        if ev == "novdec":
            row.append(rm(y, p))
    print(f"{E:>3} | {row[0]:15.2f} {row[1]:8.2f} {row[2]:8.2f} | {row[3]:15.2f} {row[4]:8.2f} {row[5]:8.2f} {row[6]:8.2f} | {np.mean(list(al.values())):.3f}")
