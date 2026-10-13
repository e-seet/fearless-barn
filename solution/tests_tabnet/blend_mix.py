#!/usr/bin/env python3
"""Does TabNet help INSIDE the shipped per-airport tree+MLP blend?

Blend form (predict.py airport_operator_multimodel): pred = w_a*tree + (1-w_a)*neural,
w_a per airport from the shipped msmlp model.json (weights solved on forward_inner
with the registered MLP; reused frozen here -- an approximation, a real build
re-solves them). tree = registered airport_operator_blend (seed 42).
  base : neural = mean of 3 random MLP (fast-trained pool, seeds 48-71)
  A    : neural = mean of those 3 + 3 other MLP     (equal-count baseline)
  B    : neural = mean of those 3 + 3 TabNet (lr2e-3p seeds 42-44)
Paired over 300 draws. Reports pooled / LIRF / non-LIRF-mix RMSE gains (A-B) and
a rough board prediction from tests_local_vs_board's LIRF model
(board^2 = sum_{a!=LIRF} W_a r_a^2 + W_LIRF (k r_LIRF)^2, k=2.05, non-LIRF=local).
Pre-registered bar: B beats A by >= 2s pooled on TEST and > 0 on VAL.
"""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
from common import dataset as D
from common.predict import predict_arm

W = json.load(open(ROOT / "artifacts/audit/audit_report.json"))["leaderboard_airport_weights"]
LS = ROOT / "tests_lirf_seeds"; m = np.load(LS / "meta.npz", allow_pickle=True)
wpa = json.load(open(ROOT / "artifacts/runs/airport_operator_multimodel__forward-with_aobt-keep_all__seed42__tag=msmlp/model.json"))["weights_per_airport"]
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2))); rng = np.random.default_rng(2); K = 2.05
Xtr, _, _ = D.load_part("forward", "train", "with_aobt", filter_target=True, outlier_policy="keep_all")

def board(y, e, idx):
    non = sum(W[a] * rm(y[idx[a]], e[idx[a]]) ** 2 for a in W if a != "LIRF")
    return float(np.sqrt(non + W["LIRF"] * (K * rm(y[idx["LIRF"]], e[idx["LIRF"]])) ** 2))

for part, pk, yk, ak in (("val", "pv", "yva", "apva"), ("test", "pt", "yte", "apte")):
    y, ap = m[yk], m[ak]
    X, _, meta = D.load_part("forward", part, "with_aobt", filter_target=False); X = D.align_columns(Xtr, X)
    tree = predict_arm("airport_operator_blend", "forward", part, "with_aobt", 42, "keep_all", X=X, meta=meta, tag=None)
    w = np.array([wpa.get(a, 0.5) for a in ap]); idx = {a: np.flatnonzero(ap == a) for a in W}
    M = np.array([np.load(LS / f"preds_seed{s}.npz")[pk] for s in range(48, 72)])
    T = np.array([np.load(HERE / f"preds_seed{s}_lr2e-3p.npz")[pk] for s in (42, 43, 44)])
    mix = lambda nn: w * tree + (1 - w) * nn
    def st(e):
        non = np.sqrt(sum(W[a] * rm(y[idx[a]], e[idx[a]]) ** 2 for a in W if a != "LIRF") / (1 - W["LIRF"]))
        return np.array([rm(y, e), rm(y[idx["LIRF"]], e[idx["LIRF"]]), non, board(y, e, idx)])
    G, SA, SB = [], [], []
    for _ in range(300):
        p = rng.permutation(24); a3 = M[p[:3]]
        A = mix(np.concatenate([a3, M[p[3:6]]]).mean(0)); B = mix((a3.sum(0) + T.sum(0)) / 6)
        sa, sb = st(A), st(B); G.append(sa - sb); SA.append(sa); SB.append(sb)
    G, SA, SB = map(np.array, (G, SA, SB))
    print(f"\n===== {part.upper()}  (shipped per-airport weights, frozen) =====")
    print("               A (blend, 6 MLP)   B (blend, 3 MLP+3 TabNet)   gain A-B (B wins)")
    for j, n in enumerate(("pooled", "LIRF", "non-LIRF mix", "pred. board")):
        print(f"  {n:13s}  {SA[:, j].mean():9.2f}          {SB[:, j].mean():9.2f}            {G[:, j].mean():+7.2f} ({(G[:, j] > 0).mean():.0%})")
    # descriptive: tabnet-only neural side
    print(f"  (descriptive) blend with TabNet-ens3 alone as neural side: pooled {rm(y, mix(T.mean(0))):.2f}; "
          f"blend with 6 MLP: {rm(y, mix(M[:6].mean(0))):.2f}")
