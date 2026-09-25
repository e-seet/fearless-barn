#!/usr/bin/env python3
"""DESCRIPTIVE sizing (no training): tree-seed averaging in the 3-member blend.

Tree side = airport_operator_blend, seed 42 (shipped) vs mean of seeds 42-47.
Weights re-solved on forward_inner for each tree side (12-seed MLP, 6-seed FTT),
frozen, applied to forward val/test. The FTT 3->6 move (which bought -1.79 on the
board) is computed on the same surface as a yardstick.
"""
import os, sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import dataset as D
from common.predict import predict_arm
from pipeline_airport_operator_catboost_multimodel.train import solve_weights_per_airport_simplex
from tests_ftt.solve_vs_strong_mlp import surface
from tests_ftt.check_6seed import blend, ftt_mean
FS, POL = "with_aobt", "keep_all"
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
F = ROOT / "tests_ftt"
fi = lambda s: F / f"preds_seed{s}_forward_inner_lr1e-3.npz"
fw = lambda s: F / (f"preds_seed{s}_lr1e-3_e132.npz" if s <= 44 else f"preds_seed{s}_lr1e-3_e175.npz")

def trees(split, part):
    f = CACHE / f"tree6_{split}_{part}.npy"
    if f.exists(): return np.load(f)
    Xtr, _, _ = D.load_part(split, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(split, part, FS, filter_target=False); X = D.align_columns(Xtr, X)
    P = np.stack([predict_arm("airport_operator_blend", split, part, FS, s, POL, X=X, meta=meta, Xtr=Xtr) for s in range(42, 48)])
    np.save(f, P); return P

yw, apw, _, mw = surface("forward_inner", "val", list(range(42, 54)))
Tw = trees("forward_inner", "val")
f6w = ftt_mean(range(42, 48), fi, "pv")
W1 = solve_weights_per_airport_simplex(yw, f6w, Tw[0], mw, apw, l2_shrinkage=0.0)
W6 = solve_weights_per_airport_simplex(yw, f6w, Tw.mean(0), mw, apw, l2_shrinkage=0.0)
f3w = ftt_mean(range(42, 45), fi, "pv")
W1f3 = solve_weights_per_airport_simplex(yw, f3w, Tw[0], mw, apw, l2_shrinkage=0.0)
print("tree weight per airport, 1 seed -> 6 seeds:", {k: f"{W1[k][1]:.2f}->{W6[k][1]:.2f}" for k in sorted(W1)})
for part, key in (("val", "pv"), ("test", "pt")):
    y, ap, _, mlp = surface("forward", part, list(range(42, 54)))
    T = trees("forward", part); ib = y <= 7200
    f6, f3 = ftt_mean(range(42, 48), fw, key), ftt_mean(range(42, 45), fw, key)
    b1 = blend(W1, f6, T[0], mlp, ap); b6 = blend(W6, f6, T.mean(0), mlp, ap)
    bf3 = blend(W1f3, f3, T[0], mlp, ap)
    seed_sd = np.std([rm(y[ib], T[s][ib]) for s in range(6)])
    print(f"forward {part}: tree standalone in-band seed42 {rm(y[ib],T[0][ib]):.2f} | ens6 {rm(y[ib],T.mean(0)[ib]):.2f} (single-seed SD {seed_sd:.2f})")
    print(f"   blend in-band: tree1 {rm(y[ib],b1[ib]):.3f} -> tree6 {rm(y[ib],b6[ib]):.3f} ({rm(y[ib],b6[ib])-rm(y[ib],b1[ib]):+.3f}) | pooled {rm(y,b1):.2f} -> {rm(y,b6):.2f}")
    print(f"   RMS move of blend: tree 1->6 {rm(b6,b1):.2f}  vs  FTT 3->6 (the -1.79 board move) {rm(b1,bf3):.2f}")
