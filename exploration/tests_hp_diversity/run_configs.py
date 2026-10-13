#!/usr/bin/env python3
"""Train mlp_antigravity-recipe variants on forward. See PREREGISTERED.md.

The fast GPU loop from tests_lirf_seeds/train_seeds.py, with the architecture
and optimizer settings taken from CONFIGS and a cap of 50 for every config.
Saves best-epoch val/test predictions in the same npz keys.

usage: python tests_hp_diversity/run_configs.py base wide deep reg bigbatch --seeds 42,...,47
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

FS, POL = "with_aobt", "keep_all"
PATIENCE = 6
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
STOP_FLAG = HERE / "STOP"

BASE = dict(hidden=(256, 128, 64), dropout=0.1, lr=1e-3, wd=1e-4, batch=2048)
CONFIGS = {
    "base": {},
    "wide": dict(hidden=(1024,)),
    "deep": dict(hidden=(128, 128, 128, 128, 64)),
    "reg": dict(dropout=0.3, wd=1e-3),
    "bigbatch": dict(batch=8192, lr=2e-3),
}


@torch.no_grad()
def predict(model, cat, num, bs=8192):
    model.eval()
    out = []
    for i in range(0, len(num), bs):
        out.append(model({c: cat[c][i:i + bs] for c in cat}, num[i:i + bs]).float().cpu().numpy())
    return np.clip(np.concatenate(out), *CLIP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+", choices=list(CONFIGS))
    ap.add_argument("--seeds", default="42,43,44,45,46,47")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--split", default="forward", choices=["forward", "seasonal_full"])
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    SPLIT, EPOCHS = a.split, a.epochs
    pre = "" if SPLIT == "forward" else f"{SPLIT}_"
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
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        card[col] = len(cats)
    med = Xtr[num_cols].astype("float64").median()
    sc = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(med).to_numpy())
    todev = lambda cn: ({c: t.to(dev) for c, t in cn[0].items()}, cn[1].to(dev))
    tr = todev(encode_data(Xtr, cat_cols, num_cols, cat_maps, sc, med))
    va = todev(encode_data(Xva, cat_cols, num_cols, cat_maps, sc, med))
    te = todev(encode_data(Xte, cat_cols, num_cols, cat_maps, sc, med))
    y_t = torch.tensor(np.asarray(ytr, dtype="float32"), device=dev)
    n = len(y_t)
    if not (HERE / f"meta{'' if SPLIT == 'forward' else '_' + SPLIT}.npz").exists():
        np.savez(HERE / f"meta{'' if SPLIT == 'forward' else '_' + SPLIT}.npz", yva=yva, yte=yte,
                 apva=mva["ADEP_mvt"].astype(str).to_numpy(),
                 apte=mte["ADEP_mvt"].astype(str).to_numpy())

    for cname in a.configs:
        cfg = {**BASE, **CONFIGS[cname]}
        for seed in seeds:
            out = HERE / f"preds_{pre}{cname}_seed{seed}.npz"
            if out.exists():
                print(f"{cname} seed {seed} exists, skip", flush=True)
                continue
            if STOP_FLAG.exists():
                print("STOP flag present -- exiting", flush=True)
                return 0
            t0 = time.time()
            seed_everything(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = TabularMLP(card, num_features=len(num_cols), hidden_dims=cfg["hidden"],
                               dropout=cfg["dropout"], prior_mean=float(np.mean(ytr))).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
            crit = nn.MSELoss()
            best, best_ep, bad, bpv, bpt, hist = 1e18, 0, 0, None, None, []
            B = cfg["batch"]
            for ep in range(1, EPOCHS + 1):
                model.train()
                perm = torch.randperm(n, device=dev)
                for i in range(0, n, B):
                    b = perm[i:i + B]
                    opt.zero_grad()
                    loss = crit(model({c: tr[0][c][b] for c in tr[0]}, tr[1][b]), y_t[b])
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    opt.step()
                pv = predict(model, *va)
                r = float(np.sqrt(np.mean((yva - pv) ** 2)))
                hist.append(r)
                if r < best:
                    best, best_ep, bad, bpv, bpt = r, ep, 0, pv, predict(model, *te)
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            hit_cap = ep >= EPOCHS and bad < PATIENCE
            np.savez(out, pv=bpv, pt=bpt, best_epoch=best_ep, val=best, epochs_run=ep,
                     hit_cap=hit_cap, history=np.asarray(hist), config=cname)
            flag = "  *** HIT CAP ***" if hit_cap else ""
            print(f"{cname} seed {seed}: best_ep {best_ep} val {best:.2f} ran {ep} "
                  f"[{time.time()-t0:.0f}s]{flag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
