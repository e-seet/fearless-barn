#!/usr/bin/env python3
"""Screen (no training): does mixing architectures beat adding MLP seeds?

Seed-averaging of the MLP plateaus at 12-18 seeds and the board pays more for
MLP variance reduction than local val shows (tests_local_vs_board). The GRU
state model (tests_seqrnn) tied the MLP standalone, so as an ensemble MEMBER
it may carry decorrelated error. Uses saved seqrnn checkpoints (seeds 42-44,
best epoch) and registered untagged mlp_antigravity forward runs.

Ensembles compared (equal-weight mean of predictions, forward val AND test):
  A  6 MLP seeds                       (baseline at equal member count)
  B  3 MLP (42-44) + 3 seqrnn (42-44)  (architecture-mixed, same count)
  C  3 MLP alone, D 3 seqrnn alone     (reference)
Also residual correlation MLP-ens vs seqrnn-ens.

Pre-registered bar (before running): B must beat A on TEST by >= 2s AND on
val by > 0, with the swapped control B' (3 MLP 45-47 + 3 seqrnn) agreeing in
sign. Anything less is closed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import features as F
from common.predict import predict_arm
from tests_seqrnn import sequence_features as SF
from tests_seqrnn.model import SeqTabularModel
from tests_seqrnn.run_seqrnn import encode_data

SPLIT, FS, POL = "forward", "with_aobt", "keep_all"
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    parts = {}
    for part in ("val", "test"):
        X, y, meta = D.load_part(SPLIT, part, FS, filter_target=False)
        parts[part] = (D.align_columns(Xtr, X), np.asarray(y, dtype="float64"), meta)

    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_maps, card = {}, {}
    for col in cat_cols:
        cats = sorted(Xtr[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        card[col] = len(cats)
    med = Xtr[num_cols].astype("float64").median()
    scaler = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(med).to_numpy())
    seq_all = SF.load_scope_cached("train2025")

    P = {"val": {}, "test": {}}
    for part, (X, y, meta) in parts.items():
        cat, num = encode_data(X, cat_cols, num_cols, cat_maps, scaler, med)
        sq = SF.align_to_ids(seq_all, X.index.to_numpy())
        seq, msk = torch.tensor(sq["seq"]), torch.tensor(sq["mask"])
        for s in (42, 43, 44):
            m = SeqTabularModel(card, num_features=len(num_cols), seq_feature_dim=SF.SEQ_FEATURE_DIM,
                                gru_hidden=32, hidden_dims=(256, 128, 64), dropout=0.1,
                                prior_mean=float(np.mean(ytr))).to(dev)
            d = config.ARTIFACTS / "runs" / f"seqrnn__forward-with_aobt-keep_all__seed{s}__tag=screen"
            m.load_state_dict(torch.load(d / "model.pt", map_location=dev)); m.eval()
            out = []
            with torch.no_grad():
                for i in range(0, len(y), 8192):
                    xc = {c: cat[c][i:i + 8192].to(dev) for c in cat}
                    out.append(m(xc, num[i:i + 8192].to(dev), seq[i:i + 8192].to(dev),
                                 msk[i:i + 8192].to(dev)).cpu().numpy())
            P[part][("seq", s)] = np.clip(np.concatenate(out), *CLIP)
        for s in range(42, 48):
            try:
                P[part][("mlp", s)] = predict_arm("mlp_antigravity", SPLIT, part, FS, s, POL,
                                                  X=X, meta=meta, tag=None)
            except Exception as e:                      # noqa: BLE001
                print(f"[{part}] mlp seed {s} unavailable: {type(e).__name__}")
    have = sorted(s for (a, s) in P["val"] if a == "mlp")
    print("MLP seeds available:", have)

    def ens(part, keys):
        return np.mean([P[part][k] for k in keys], axis=0)

    mlp = lambda ss: [("mlp", s) for s in ss]
    seq = lambda ss: [("seq", s) for s in ss]
    sets = {
        "single mlp (mean of seeds)": None,
        "single seqrnn (mean of seeds)": None,
        "C  3 MLP 42-44": mlp([42, 43, 44]),
        "D  3 seqrnn 42-44": seq([42, 43, 44]),
        "A  6 MLP": mlp(have[:6]) if len(have) >= 6 else None,
        "B  3 MLP 42-44 + 3 seqrnn": mlp([42, 43, 44]) + seq([42, 43, 44]),
        "B' 3 MLP 45-47 + 3 seqrnn": (mlp([45, 46, 47]) + seq([42, 43, 44])) if len(have) >= 6 else None,
        "A' 3 MLP 45-47": mlp([45, 46, 47]) if len(have) >= 6 else None,
    }
    for part in ("val", "test"):
        y = parts[part][1]
        print(f"\n--- {part.upper()} ---")
        print("  single mlp   per seed:", " ".join(f"{rm(y, P[part][('mlp', s)]):7.2f}" for s in have))
        print("  single seqrnn per seed:", " ".join(f"{rm(y, P[part][('seq', s)]):7.2f}" for s in (42, 43, 44)))
        for n, k in sets.items():
            if k:
                print(f"  {n:30s} {rm(y, ens(part, k)):8.2f}")
        em, es = ens(part, mlp([42, 43, 44])), ens(part, seq([42, 43, 44]))
        print(f"  corr(resid MLP-ens3, seqrnn-ens3) = {np.corrcoef(y - em, y - es)[0, 1]:.4f}")
    for part in ("val", "test"):
        y = parts[part][1]
        if sets["A  6 MLP"]:
            a, b = rm(y, ens(part, sets["A  6 MLP"])), rm(y, ens(part, sets["B  3 MLP 42-44 + 3 seqrnn"]))
            print(f"\n{part}: B vs A gain {a - b:+.2f}s", end="")
            if sets["B' 3 MLP 45-47 + 3 seqrnn"]:
                a2, b2 = rm(y, ens(part, mlp(have[:6]))), rm(y, ens(part, sets["B' 3 MLP 45-47 + 3 seqrnn"]))
                print(f"   B' vs A gain {a2 - b2:+.2f}s")
            else:
                print()


if __name__ == "__main__":
    main()
