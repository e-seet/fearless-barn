#!/usr/bin/env python3
"""Second-surface confirmation of the TabNet-at-LIRF lead on seasonal_full.

seasonal_full: train Feb-Jun + Aug-Nov 2025, val Dec 2025, test Jan + Jul 2025,
all 10 airports in both test months (the scored-set composition). Compares, on
the SAME registered arms the shipped blend uses:
  A    = blend(tree, neural = mean of registered MLP seeds 42-47)
  B_i  = blend(tree, neural = mean of 3 registered MLP (one of the 20 triples) +
                              3 TabNet seeds 42-44 (sf_lr2e-3p))
Blend = w_a*tree + (1-w_a)*neural with the SHIPPED per-airport weights (msmlp,
frozen; seasonal_full is selection-forbidden anyway). tree = registered
airport_operator_blend seasonal_full seed 42.

Pre-registered bars (before any seasonal_full TabNet number was seen):
 CONFIRM (strong): on TEST, B beats A pooled by >= 2s AND LIRF by > 0 in >= 80%
   of the 20 triples, AND Dec-val pooled gain > 0.
 CONFIRM (LIRF mechanism only): LIRF gain >= 5s in >= 80% of triples on TEST
   even if pooled < 2s (forward result was pooled +1.70 / LIRF +10.2).
 Plain-ensemble check: residual corr(TabNet-ens3, MLP-ens3) < 0.95.
 Anything else = not confirmed; the forward result then stays a single-surface
 lead and nothing is built.
Rough board prediction uses tests_local_vs_board's LIRF model (k=2.05).
"""
import itertools, json, sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from common import dataset as D
from common.predict import predict_arm

SPLIT = "seasonal_full"
W = json.load(open(ROOT / "artifacts/audit/audit_report.json"))["leaderboard_airport_weights"]
wpa = json.load(open(ROOT / "artifacts/runs/airport_operator_multimodel__forward-with_aobt-keep_all__seed42__tag=msmlp/model.json"))["weights_per_airport"]
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2))); K = 2.05
Xtr, _, _ = D.load_part(SPLIT, "train", "with_aobt", filter_target=True, outlier_policy="keep_all")

def board(y, e, idx):
    non = sum(W[a] * rm(y[idx[a]], e[idx[a]]) ** 2 for a in W if a != "LIRF")
    return float(np.sqrt(non + W["LIRF"] * (K * rm(y[idx["LIRF"]], e[idx["LIRF"]])) ** 2))

for part, pk in (("val", "pv"), ("test", "pt")):
    X, y, meta = D.load_part(SPLIT, part, "with_aobt", filter_target=False); X = D.align_columns(Xtr, X)
    y = np.asarray(y, float); ap = meta["ADEP_mvt"].astype(str).to_numpy()
    idx = {a: np.flatnonzero(ap == a) for a in W}
    month = meta["month"].to_numpy() if "month" in meta.columns else None
    tree = predict_arm("airport_operator_blend", SPLIT, part, "with_aobt", 42, "keep_all", X=X, meta=meta, tag=None)
    M = np.array([predict_arm("mlp_antigravity", SPLIT, part, "with_aobt", s, "keep_all", X=X, meta=meta, tag=None)
                  for s in range(42, 48)])
    T = np.array([np.load(HERE / f"preds_seed{s}_sf_lr2e-3p.npz")[pk] for s in (42, 43, 44)])
    w = np.array([wpa.get(a, 0.5) for a in ap]); mix = lambda nn: w * tree + (1 - w) * nn
    def st(e):
        non = np.sqrt(sum(W[a] * rm(y[idx[a]], e[idx[a]]) ** 2 for a in W if a != "LIRF") / (1 - W["LIRF"]))
        return np.array([rm(y, e), rm(y[idx["LIRF"]], e[idx["LIRF"]]), non, board(y, e, idx)])
    print(f"\n===== {SPLIT} {part.upper()}  n={len(y)} =====")
    print("MLP single seeds pooled:", " ".join(f"{rm(y, p):.1f}" for p in M),
          "| TabNet single:", " ".join(f"{rm(y, t):.1f}" for t in T), f"| TabNet-ens3 {rm(y, T.mean(0)):.2f} | MLP-ens6 {rm(y, M.mean(0)):.2f}")
    trip = list(itertools.combinations(range(6), 3))
    c = [np.corrcoef(y - T.mean(0), y - M[list(t)].mean(0))[0, 1] for t in trip]
    cm = [np.corrcoef(y - M[list(t)].mean(0), y - M[[i for i in range(6) if i not in t]].mean(0))[0, 1] for t in trip]
    print(f"residual corr TabNet-ens3 vs MLP-ens3 = {np.mean(c):.4f}   (MLP-ens3 vs complementary MLP-ens3 = {np.mean(cm):.4f})")
    SA = st(mix(M.mean(0)))
    G = np.array([SA - st(mix((M[list(t)].sum(0) + T.sum(0)) / 6)) for t in trip])
    print(f"                  A(blend,6 MLP)  gain A-B mean [min,max]           B wins")
    for j, n in enumerate(("pooled", "LIRF", "non-LIRF mix", "pred. board")):
        print(f"  {n:13s} {SA[j]:10.2f}    {G[:, j].mean():+7.2f} [{G[:, j].min():+6.2f},{G[:, j].max():+6.2f}]   {(G[:, j] > 0).mean():.0%}")
    if part == "test":
        print("  per-airport RMSE, A vs B(mean of triples):")
        Bm = np.mean([mix((M[list(t)].sum(0) + T.sum(0)) / 6) for t in trip], axis=0); Am = mix(M.mean(0))
        for a in sorted(W, key=lambda a: -W[a]):
            print(f"    {a}: A {rm(y[idx[a]], Am[idx[a]]):7.1f}  B {rm(y[idx[a]], Bm[idx[a]]):7.1f}  gain {rm(y[idx[a]], Am[idx[a]]) - rm(y[idx[a]], Bm[idx[a]]):+6.1f}")
        if month is not None:
            for mo in np.unique(month):
                ix = np.flatnonzero(month == mo); lx = np.flatnonzero((month == mo) & (ap == "LIRF"))
                print(f"    month {mo}: pooled A {rm(y[ix], Am[ix]):.1f} B {rm(y[ix], Bm[ix]):.1f} | LIRF A {rm(y[lx], Am[lx]):.1f} B {rm(y[lx], Bm[lx]):.1f}")
