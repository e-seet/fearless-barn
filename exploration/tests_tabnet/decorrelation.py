#!/usr/bin/env python3
"""TabNet decorrelation / ensemble-member check (no training; uses saved preds).

Question: converged TabNet (lr 2e-3, tests_tabnet) ties the MLP standalone but
is a different inductive bias. Are its errors decorrelated enough from the MLP's
to beat adding more MLP seeds? (The GRU state model was r=0.98 and failed.)

Pool: 24 fast-trained MLP seeds (tests_lirf_seeds/preds_seed48..71, same recipe
as the registered arm) + 3 TabNet seeds (preds_seed{42,43,44}_lr2e-3p).
All on forward val / test, equal-weight means.

  A  = 3 random MLP + 3 OTHER random MLP   (equal member count baseline)
  B  = the same first 3 MLP + the 3 TabNet
Paired over 300 random draws (the first 3 MLP are shared, so the only
difference is 3 TabNet vs 3 more MLP).

Pre-registered bar (before running): mean gain (A - B, positive = TabNet
better) >= 2s on TEST pooled AND > 0 on VAL AND B beats A in >= 80% of draws
on VAL; residual corr(TabNet-ens3, MLP-ens3) reported (< 0.95 needed, GRU was
0.98). LIRF and non-LIRF reported separately because the board is LIRF-driven
(tests_local_vs_board). Weight sweep is descriptive only, not selection.
"""
import json
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
LS = HERE.parent / "tests_lirf_seeds"
W = json.load(open(HERE.parent / "artifacts/audit/audit_report.json"))["leaderboard_airport_weights"]
m = np.load(LS / "meta.npz", allow_pickle=True)
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
rng = np.random.default_rng(0)
TAB_TAG = "lr2e-3p"

for part, pk, yk, ak in (("val", "pv", "yva", "apva"), ("test", "pt", "yte", "apte")):
    y, ap = m[yk], m[ak]
    L = np.flatnonzero(ap == "LIRF")
    idx = {a: np.flatnonzero(ap == a) for a in W}
    M = np.array([np.load(LS / f"preds_seed{s}.npz")[pk] for s in range(48, 72)])
    T = np.array([np.load(HERE / f"preds_seed{s}_{TAB_TAG}.npz")[pk] for s in (42, 43, 44)])
    tab = T.mean(0)

    def stats(e):
        non = np.sqrt(sum(W[a] * rm(y[idx[a]], e[idx[a]]) ** 2 for a in W if a != "LIRF") / (1 - W["LIRF"]))
        return rm(y, e), rm(y[L], e[L]), non

    print(f"\n===== {part.upper()} =====")
    print("TabNet single seeds pooled:", " ".join(f"{rm(y, t):.2f}" for t in T),
          f"| TabNet-ens3 pooled {rm(y, tab):.2f}")
    G, C_tm, C_mm = [], [], []
    for _ in range(300):
        p = rng.permutation(24)
        a3, b3 = M[p[:3]].mean(0), M[p[3:6]].mean(0)
        A = (a3 + b3) / 2
        B = (a3 * 3 + T.sum(0)) / 6
        G.append(np.array(stats(A)) - np.array(stats(B)))
        C_tm.append(np.corrcoef(y - a3, y - tab)[0, 1])
        C_mm.append(np.corrcoef(y - a3, y - b3)[0, 1])
    G = np.array(G)
    print(f"residual corr: TabNet-ens3 vs MLP-ens3 = {np.mean(C_tm):.4f}   (MLP-ens3 vs another MLP-ens3 = {np.mean(C_mm):.4f})")
    for j, n in enumerate(("pooled", "LIRF", "non-LIRF mix")):
        print(f"  gain A-B ({n:12s}): mean {G[:, j].mean():+.3f}  sd {G[:, j].std():.3f}  "
              f"B better in {(G[:, j] > 0).mean():.0%} of draws")
    print("  descriptive weight sweep, w*TabNet-ens3 + (1-w)*MLP-ens6 (first 6 pool seeds): "
          + "  ".join(f"w={w}: {rm(y, w * tab + (1 - w) * M[:6].mean(0)):.2f}" for w in (0, .1, .2, .3, .5)))
