#!/usr/bin/env python3
"""FT-Transformer decorrelation / ensemble-member screen (no training).

Same construction as tests_tabnet/decorrelation.py -- deliberately, so the
number is comparable to the one that predicted TabNet's outcome (0.88-0.94 ->
paid) and the GRU's (0.980 -> dead).

Pool: 24 MLP seeds (tests_lirf_seeds/preds_seed48..71, the registered
mlp_antigravity recipe) + the FT-Transformer seeds trained by run_ftt.py.
All on forward val / test, equal-weight means.

  A = 3 random MLP + 3 OTHER random MLP   (equal member count baseline)
  B = the same first 3 MLP + the 3 FT-Transformer seeds

Paired over 300 random draws; the first 3 MLP are shared, so the only
difference is 3 FTT vs 3 more MLP.

The bar is in PREREGISTERED.md and was fixed before any run. This script only
reports; it does not select.

usage: python tests_ftt/decorrelation.py [--tag TAG] [--seeds 42 43 44]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LS = ROOT / "tests_lirf_seeds"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="")
ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
a = ap.parse_args()
suffix = f"_{a.tag}" if a.tag else ""

W = json.load(open(ROOT / "artifacts/audit/audit_report.json"))["leaderboard_airport_weights"]
m = np.load(LS / "meta.npz", allow_pickle=True)
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
rng = np.random.default_rng(0)

# Convergence gate (PREREGISTERED.md): a seed that ended AT the cap is
# truncated, not converged, and must not enter the screen.
Z = {}
for s in a.seeds:
    p = HERE / f"preds_seed{s}{suffix}.npz"
    if not p.exists():
        sys.exit(f"missing {p} -- run: python tests_ftt/run_ftt.py {' '.join(map(str, a.seeds))}")
    z = np.load(p, allow_pickle=True)
    if bool(z["hit_cap"]):
        sys.exit(f"seed {s} HIT THE EPOCH CAP (best {int(z['best_epoch'])} of "
                 f"{int(z['epochs_run'])}) -- truncated, not converged. Raise "
                 f"--epochs to >= 1.5x the observed best and re-run it before screening.")
    Z[s] = z
print("FTT seeds:", {s: (int(Z[s]["best_epoch"]), round(float(Z[s]["val"]), 2)) for s in a.seeds})
print(f"params: {float(Z[a.seeds[0]]['n_params'])/1e6:.2f}M   lr: {float(Z[a.seeds[0]]['lr']):.1e}")

for part, pk, yk, ak in (("val", "pv", "yva", "apva"), ("test", "pt", "yte", "apte")):
    y, ap_arr = m[yk], m[ak]
    L = np.flatnonzero(ap_arr == "LIRF")
    idx = {x: np.flatnonzero(ap_arr == x) for x in W}
    M = np.array([np.load(LS / f"preds_seed{s}.npz")[pk] for s in range(48, 72)])
    T = np.array([Z[s][pk] for s in a.seeds])
    ftt = T.mean(0)

    def stats(e):
        non = np.sqrt(sum(W[x] * rm(y[idx[x]], e[idx[x]]) ** 2
                          for x in W if x != "LIRF") / (1 - W["LIRF"]))
        return rm(y, e), rm(y[L], e[L]), non

    print(f"\n===== {part.upper()} =====")
    print("FTT single seeds pooled:", " ".join(f"{rm(y, t):.2f}" for t in T),
          f"| FTT-ens{len(T)} pooled {rm(y, ftt):.2f}",
          f"| MLP-ens3 pooled {rm(y, M[:3].mean(0)):.2f}")
    G, C_fm, C_mm = [], [], []
    for _ in range(300):
        p = rng.permutation(24)
        a3, b3 = M[p[:3]].mean(0), M[p[3:6]].mean(0)
        A = (a3 + b3) / 2
        B = (a3 * 3 + T.sum(0)) / (3 + len(T))
        G.append(np.array(stats(A)) - np.array(stats(B)))
        C_fm.append(np.corrcoef(y - a3, y - ftt)[0, 1])
        C_mm.append(np.corrcoef(y - a3, y - b3)[0, 1])
    G = np.array(G)
    c_fm = float(np.mean(C_fm))
    print(f"residual corr: FTT-ens{len(T)} vs MLP-ens3 = {c_fm:.4f}   "
          f"(MLP-ens3 vs another MLP-ens3 = {np.mean(C_mm):.4f})")
    print(f"   reference: GRU 0.980 (dead), router 0.898/0.876 (dead), "
          f"TabNet 0.88-0.94 (paid)  -> bar is < 0.95")
    for j, nm in enumerate(("pooled", "LIRF", "non-LIRF mix")):
        print(f"  gain A-B ({nm:12s}): mean {G[:, j].mean():+.3f}  sd {G[:, j].std():.3f}  "
              f"B better in {(G[:, j] > 0).mean():.0%} of draws")
    if part == "test":
        ok = (c_fm < 0.95) and (G[:, 0].mean() >= 2.0)
        print(f"\nTEST-side bar (corr < 0.95 AND pooled gain >= 2s): "
              f"{'PASS' if ok else 'FAIL'}")
    else:
        ok = (c_fm < 0.95) and (G[:, 0].mean() > 0) and ((G[:, 0] > 0).mean() >= 0.80)
        print(f"\nVAL-side bar (corr < 0.95 AND gain > 0 AND B better in >= 80%): "
              f"{'PASS' if ok else 'FAIL'}")
