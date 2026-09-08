#!/usr/bin/env python3
"""Arm mlp_antigravity: PyTorch Tabular Neural Network with Entity Embeddings.

Provides a fundamentally different inductive bias (deep continuous + embedding space)
to pair against decision trees (LightGBM & XGBoost).
Runs on NVIDIA CUDA with batch normalization, AdamW, and early stopping on val RMSE.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import dataset as D
from common import features as F
from common.metrics import rmse
from common.bootstrap import (convergence_evidence, declare_selection_set, run_dir,
                              seed_everything, write_json_guarded)
from pipeline_mlp_antigravity.model import TabularMLP

ARM = "mlp_antigravity"


class TabularDataset(Dataset):
    def __init__(self, cat_tensors: dict[str, torch.Tensor], num_tensor: torch.Tensor,
                 targets: torch.Tensor = None):
        self.cat_tensors = cat_tensors
        self.num_tensor = num_tensor
        self.targets = targets

    def __len__(self) -> int:
        return len(self.num_tensor)

    def __getitem__(self, idx: int):
        x_cat = {col: self.cat_tensors[col][idx] for col in self.cat_tensors}
        x_num = self.num_tensor[idx]
        if self.targets is not None:
            return x_cat, x_num, self.targets[idx]
        return x_cat, x_num


def encode_data(X: pd.DataFrame, cat_cols: list[str], num_cols: list[str],
                cat_maps: dict[str, dict], scaler: StandardScaler,
                medians: pd.Series):
    cat_tensors = {}
    for col in cat_cols:
        m = cat_maps[col]
        unk = len(m)
        vals = X[col].astype("object").map(m).fillna(unk).to_numpy(dtype="int64")
        cat_tensors[col] = torch.tensor(vals, dtype=torch.long)

    num_vals = X[num_cols].astype("float64").fillna(medians).to_numpy()
    num_scaled = scaler.transform(num_vals)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)
    return cat_tensors, num_tensor


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--epochs", type=int, default=25,
                    help="maximum training epochs")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--freeze-from", default=None)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed)

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                       filter_target=True,
                                       outlier_policy=args.outlier_policy)
    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]

    refit_only = not config.SPLITS[args.split]["val"]
    frozen = None
    if refit_only:
        if not args.freeze_from:
            raise SystemExit("split '{}' has no val. Pass --freeze-from.".format(args.split))
        frozen = json.loads(Path(args.freeze_from).read_text())
        n_epochs = frozen.get("best_epoch", 15)
        print("[{}] REFIT on '{}' for {} epochs (frozen from {})".format(
            ARM, args.split, n_epochs, args.freeze_from))
    else:
        Xva, yva, _ = D.load_part(args.split, "val", args.featureset, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
        n_epochs = args.epochs

    print("[{}] train n={} val n={} cat_feats={} num_feats={} device={}".format(
        ARM, len(Xtr), 0 if refit_only else len(Xva), len(cat_cols), len(num_cols), device))

    # Preprocessing
    cat_maps = {}
    cat_cardinalities = {}
    for col in cat_cols:
        cats = sorted(Xtr[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        cat_cardinalities[col] = len(cats)

    medians = Xtr[num_cols].astype("float64").median()
    scaler = StandardScaler()
    scaler.fit(Xtr[num_cols].astype("float64").fillna(medians).to_numpy())

    prior_mean = float(ytr.mean())
    with open(out / "preprocessor.pkl", "wb") as fh:
        pickle.dump({
            "cat_cols": cat_cols, "num_cols": num_cols,
            "cat_maps": cat_maps, "cat_cardinalities": cat_cardinalities,
            "scaler": scaler, "medians": medians, "prior_mean": prior_mean
        }, fh)

    # Prepare DataLoaders
    tr_cat_tensors, tr_num_tensor = encode_data(Xtr, cat_cols, num_cols, cat_maps, scaler, medians)
    tr_targets = torch.tensor(np.asarray(ytr, dtype="float32"))
    tr_ds = TabularDataset(tr_cat_tensors, tr_num_tensor, tr_targets)
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)

    if not refit_only:
        va_cat_tensors, va_num_tensor = encode_data(Xva, cat_cols, num_cols, cat_maps, scaler, medians)
        va_targets = torch.tensor(np.asarray(yva, dtype="float32"))
        va_ds = TabularDataset(va_cat_tensors, va_num_tensor, va_targets)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=0)

    model = TabularMLP(cat_cardinalities, num_features=len(num_cols), prior_mean=prior_mean).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_epoch = 0
    best_val_rmse = float("inf")
    patience_cnt = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_samples = 0

        for x_cat, x_num, y_b in tr_loader:
            x_cat = {c: x_cat[c].to(device) for c in x_cat}
            x_num = x_num.to(device)
            y_b = y_b.to(device)

            optimizer.zero_grad()
            preds = model(x_cat, x_num)
            loss = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            n_samples += len(y_b)

        tr_rmse = np.sqrt(total_loss / n_samples)

        if refit_only:
            print("  Epoch {:2d}/{} [{:.1f}s] train RMSE={:.3f}".format(
                epoch, n_epochs, time.time() - t0, tr_rmse))
            torch.save(model.state_dict(), out / "model.pt")
            best_epoch = epoch
        else:
            model.eval()
            va_preds = []
            with torch.no_grad():
                for x_cat, x_num, _ in va_loader:
                    x_cat = {c: x_cat[c].to(device) for c in x_cat}
                    x_num = x_num.to(device)
                    p = model(x_cat, x_num).cpu().numpy()
                    va_preds.append(p)

            va_preds = np.concatenate(va_preds)
            va_preds = np.clip(va_preds, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
            va_rmse = rmse(yva, va_preds)
            history.append({"epoch": epoch, "val_rmse": float(va_rmse)})

            print("  Epoch {:2d}/{} [{:.1f}s] train RMSE={:.3f} | val RMSE={:.3f}".format(
                epoch, n_epochs, time.time() - t0, tr_rmse, va_rmse))

            if va_rmse < best_val_rmse:
                best_val_rmse = va_rmse
                best_epoch = epoch
                patience_cnt = 0
                torch.save(model.state_dict(), out / "model.pt")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print("  Early stopping fired after epoch {} (best epoch {})".format(epoch, best_epoch))
                    break

    sel = declare_selection_set("mlp_antigravity best_epoch",
                                selected_on="frozen" if refit_only else "val",
                                value=int(best_epoch))

    conv = convergence_evidence(history, max_epochs=args.epochs, patience=args.patience,
                                metric_key="val_rmse", mode="min") if not refit_only else {
        "refit_only": True, "epochs": best_epoch}

    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "best_epoch": int(best_epoch),
        "best_iteration": int(best_epoch),
        "val_rmse_clipped": float(best_val_rmse) if not refit_only else None,
        "refit_only": bool(refit_only),
        "convergence": conv, "selection": dict(sel),
        "n_train": int(len(Xtr)), "n_features": int(Xtr.shape[1]),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "outlier_policy": dict(_pool_meta.attrs.get("outlier_policy", {})),
    }, force=args.force)

    print("[{}] finished -> val RMSE: {:.3f}".format(ARM, best_val_rmse if not refit_only else 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
