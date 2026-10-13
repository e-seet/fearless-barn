#!/usr/bin/env python3
"""Convergence checks + snapshot-averaging contrasts from preds_seed*.npz."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import os
EP = int(os.environ.get("SNAP_EPOCHS", 60))
LO = int(os.environ.get("SNAP_LO", 41))   # 1-based first epoch of averaging window
SEEDS = [int(a) for a in sys.argv[1:]] or [42, 43, 44]


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def load(seed):
    z = np.load(HERE / f"preds_e{EP}_seed{seed}.npz")
    return {k: z[k] for k in z.files}


runs = {s: load(s) for s in SEEDS}
E = runs[SEEDS[0]]["pv"].shape[0]

print("=" * 78)
print("CONVERGENCE CHECKS (per seed)")
print("=" * 78)
conv_ok = True
for s, r in runs.items():
    yv = r["yva"]
    v = np.array([rmse(yv, p) for p in r["pv"]])
    best = int(v.argmin()) + 1
    early = v[LO - 1:LO + 4].mean()          # first 5 epochs of window
    late = v[E - 5:].mean()                   # last 5 epochs
    slope = np.polyfit(np.arange(LO, E + 1), v[LO - 1:], 1)[0]
    win_sd = v[LO - 1:].std(ddof=1)
    c1 = best <= E - 2
    c2 = (early - late) <= 2.0
    ok = c1 and c2
    conv_ok &= ok
    print(f"seed {s}: val curve ep1..{E}: " + " ".join(f"{x:.0f}" for x in v))
    print(f"   best epoch {best}/{E} (pinned at cap? {not c1})  "
          f"mean win-first5 {early:.2f}  mean last5 {late:.2f}  "
          f"drift(early-late) {early-late:+.2f}  window slope {slope:+.3f}s/epoch  "
          f"window sd {win_sd:.2f}  -> {'CONVERGED' if ok else 'NOT CONVERGED'}")
print(f"\nALL SEEDS CONVERGED: {conv_ok}\n")

print("=" * 78)
print("CONTRASTS (paired within run)")
print("=" * 78)
names = ["best", "avg13", "avg21", "swa13"]


def variants(r, part):
    P = r["pv"] if part == "val" else r["pt"]
    y = r["yva"] if part == "val" else r["yte"]
    yv = r["yva"]
    bi = int(np.argmin([rmse(yv, p) for p in r["pv"]]))   # val-selected epoch
    return y, {
        "best": P[bi],
        "avg13": P[LO - 1:].mean(0),
        "avg21": P[E - 5:].mean(0),
        "swa13": r["swa_va"] if part == "val" else r["swa_te"],
    }


for part in ("val", "test"):
    tab = {n: [] for n in names}
    ens = {n: [] for n in names}
    for s, r in runs.items():
        y, vs = variants(r, part)
        for n in names:
            tab[n].append(rmse(y, vs[n]))
            ens[n].append(vs[n])
    print(f"\n--- {part.upper()} single-model RMSE per seed ---")
    for n in names:
        print(f"  {n:6s} " + "  ".join(f"{x:8.2f}" for x in tab[n]) +
              f"   mean {np.mean(tab[n]):8.2f}  sd {np.std(tab[n], ddof=1):5.2f}")
    print(f"  paired gain vs best (positive = better), per seed and mean:")
    for n in names[1:]:
        g = np.array(tab["best"]) - np.array(tab[n])
        print(f"  {n:6s} " + "  ".join(f"{x:+8.2f}" for x in g) +
              f"   mean {g.mean():+7.2f}   sign-consistent {bool((g > 0).all())}")
    y = runs[SEEDS[0]]["yva" if part == "val" else "yte"]
    print(f"  {len(SEEDS)}-seed ENSEMBLE RMSE (mean of seeds' predictions):")
    base = rmse(y, np.mean(ens["best"], axis=0))
    for n in names:
        e = rmse(y, np.mean(ens[n], axis=0))
        print(f"  {n:6s} {e:8.2f}   vs ensemble(best) {base - e:+6.2f}")
