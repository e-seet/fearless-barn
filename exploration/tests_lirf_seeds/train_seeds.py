#!/usr/bin/env python3
"""Train extra mlp_antigravity-recipe seeds (fast GPU loop) and store best-epoch
val/test predictions, to measure the LIRF-specific seed-averaging curve.

Why LIRF specifically: tests_local_vs_board/per_airport_split.txt shows the
arms differ almost only at LIRF and the board weights LIRF ~2x its local RMSE
(k~2.0), so the pooled seed curve (plateaued at 12-18 seeds) is dominated by
non-LIRF airports and may hide a LIRF curve that is still falling. The shipped
blends give LIRF 100% MLP weight (alpha 0.00), so MLP-ensemble LIRF RMSE is
the blend's LIRF RMSE.

Recipe = pipeline_mlp_antigravity defaults: TabularMLP (256,128,64), dropout
0.1, AdamW 1e-3 wd 1e-4, batch 2048, MSE, clip 5, early stopping patience 6 on
pooled val RMSE, max 25 epochs, best-epoch checkpoint. Only difference: GPU
tensor minibatching (verified equivalent in tests_snapshot_avg: val mean
257.08 vs registered 257.01).

usage: python train_seeds.py 48 49 50 ...
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common import features as F
from common.bootstrap import seed_everything
from pipeline_mlp_antigravity.model import TabularMLP
from pipeline_mlp_antigravity.train import encode_data

SPLIT, FS, POL = "forward", "with_aobt", "keep_all"
EPOCHS, PATIENCE, BATCH = 25, 6, 2048
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


@torch.no_grad()
def predict(model, cat, num, bs=8192):
    model.eval(); out = []
    for i in range(0, len(num), bs):
        out.append(model({c: cat[c][i:i + bs] for c in cat}, num[i:i + bs]).float().cpu().numpy())
    return np.clip(np.concatenate(out), *CLIP)


def main(seeds):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    Xva, yva, mva = D.load_part(SPLIT, "val", FS, filter_target=False)
    Xte, yte, mte = D.load_part(SPLIT, "test", FS, filter_target=False)
    Xva, Xte = D.align_columns(Xtr, Xva), D.align_columns(Xtr, Xte)
    yva, yte = np.asarray(yva, float), np.asarray(yte, float)
    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_maps, card = {}, {}
    for col in cat_cols:
        cats = sorted(Xtr[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}; card[col] = len(cats)
    med = Xtr[num_cols].astype("float64").median()
    sc = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(med).to_numpy())
    todev = lambda cn: ({c: t.to(dev) for c, t in cn[0].items()}, cn[1].to(dev))
    tr = todev(encode_data(Xtr, cat_cols, num_cols, cat_maps, sc, med))
    va = todev(encode_data(Xva, cat_cols, num_cols, cat_maps, sc, med))
    te = todev(encode_data(Xte, cat_cols, num_cols, cat_maps, sc, med))
    y_t = torch.tensor(np.asarray(ytr, dtype="float32"), device=dev); n = len(y_t)
    np.savez(HERE / "meta.npz", yva=yva, yte=yte,
             apva=mva["ADEP_mvt"].astype(str).to_numpy(), apte=mte["ADEP_mvt"].astype(str).to_numpy())
    for seed in seeds:
        out = HERE / f"preds_seed{seed}.npz"
        if out.exists():
            print(f"seed {seed} exists, skip", flush=True); continue
        t0 = time.time(); seed_everything(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = TabularMLP(card, num_features=len(num_cols), hidden_dims=(256, 128, 64),
                           dropout=0.1, prior_mean=float(np.mean(ytr))).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); crit = nn.MSELoss()
        best, best_ep, bad, bpv, bpt = 1e18, 0, 0, None, None
        for ep in range(1, EPOCHS + 1):
            model.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, BATCH):
                b = perm[i:i + BATCH]; opt.zero_grad()
                loss = crit(model({c: tr[0][c][b] for c in tr[0]}, tr[1][b]), y_t[b]); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); opt.step()
            pv = predict(model, *va); r = float(np.sqrt(np.mean((yva - pv) ** 2)))
            if r < best:
                best, best_ep, bad, bpv, bpt = r, ep, 0, pv, predict(model, *te)
            else:
                bad += 1
                if bad >= PATIENCE: break
        np.savez(out, pv=bpv, pt=bpt, best_epoch=best_ep, val=best, epochs_run=ep)
        print(f"seed {seed}: best_ep {best_ep} val {best:.2f} epochs_run {ep} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
