#!/usr/bin/env python3
"""Arm per_airport_mlp: One PyTorch Tabular Neural Network per departure airport.

Why Per-Airport Neural Networks:
--------------------------------
Just as LightGBM gained 20+ seconds by transitioning from pooled to per_airport,
each airport has a completely independent physical taxiway geometry.
By training dedicated neural networks per airport, each hub receives its own
stand, runway, and airline entity embeddings without interference from other airports.
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
from common import evaluate as E
from common.metrics import rmse
from common.bootstrap import (run_dir, seed_everything, write_json_guarded)
from pipeline_per_airport.model import AIRPORT_COL
from pipeline_per_airport_mlp.model import AirportTabularMLP

ARM = "per_airport_mlp"


class TabularAirportDataset(Dataset):
    def __init__(self, cat_tensor: torch.Tensor, num_tensor: torch.Tensor,
                 targets: torch.Tensor = None):
        self.cat_tensor = cat_tensor
        self.num_tensor = num_tensor
        self.targets = targets

    def __len__(self) -> int:
        return len(self.num_tensor)

    def __getitem__(self, idx: int):
        if self.targets is not None:
            return self.cat_tensor[idx], self.num_tensor[idx], self.targets[idx]
        return self.cat_tensor[idx], self.num_tensor[idx]


def encode_airport_data(X: pd.DataFrame, cat_cols: list[str], num_cols: list[str],
                        cat_maps: dict[str, dict], scaler: StandardScaler,
                        medians: pd.Series):
    cat_vals = np.empty((len(X), len(cat_cols)), dtype=np.int64)
    for j, col in enumerate(cat_cols):
        m = cat_maps[col]
        unk = len(m)
        cat_vals[:, j] = X[col].astype("object").map(m).fillna(unk).to_numpy(dtype="int64")
    cat_tensor = torch.tensor(cat_vals, dtype=torch.long)

    num_vals = X[num_cols].astype("float64").fillna(medians).fillna(0.0).to_numpy()
    num_scaled = scaler.transform(num_vals)
    num_tensor = torch.tensor(num_scaled, dtype=torch.float32)
    return cat_tensor, num_tensor


def train_single_airport(
    a: str,
    Xtr_ap: pd.DataFrame, ytr_ap: pd.Series,
    Xva_ap: pd.DataFrame, yva_ap: pd.Series,
    cat_cols: list[str], num_cols: list[str],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    patience: int = 5,
    seed: int = 42
) -> tuple[nn.Module, dict, float, np.ndarray]:
    torch.manual_seed(seed)

    # Preprocessing maps for this airport
    cat_maps = {}
    cat_cardinalities = {}
    for col in cat_cols:
        cats = sorted(Xtr_ap[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        cat_cardinalities[col] = len(cats)

    medians = Xtr_ap[num_cols].astype("float64").median().fillna(0.0)
    scaler = StandardScaler()
    scaler.fit(Xtr_ap[num_cols].astype("float64").fillna(medians).fillna(0.0).to_numpy())

    prep = {
        "cat_cols": cat_cols, "num_cols": num_cols,
        "cat_maps": cat_maps, "cat_cardinalities": cat_cardinalities,
        "scaler": scaler, "medians": medians,
        "prior_mean": float(ytr_ap.mean())
    }

    cat_tr, num_tr = encode_airport_data(Xtr_ap, cat_cols, num_cols, cat_maps, scaler, medians)
    targets_tr = torch.tensor(ytr_ap.to_numpy(), dtype=torch.float32)
    cat_va, num_va = encode_airport_data(Xva_ap, cat_cols, num_cols, cat_maps, scaler, medians)
    targets_va = torch.tensor(yva_ap.to_numpy(), dtype=torch.float32)

    model = AirportTabularMLP(cat_cardinalities, num_features=len(num_cols),
                              hidden_dims=(128, 64), dropout=0.1, prior_mean=prep["prior_mean"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_rmse = float("inf")
    best_weights = None
    best_preds = None
    epochs_no_improve = 0
    n_tr = len(targets_tr)
    n_va = len(targets_va)

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        for b_start in range(0, n_tr, batch_size):
            b_idx = perm[b_start: b_start + batch_size]
            x_cat = cat_tr[b_idx].to(device, non_blocking=True)
            x_num = num_tr[b_idx].to(device, non_blocking=True)
            y_batch = targets_tr[b_idx].to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(x_cat, x_num)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for b_start in range(0, n_va, batch_size * 2):
                x_cat = cat_va[b_start: b_start + batch_size * 2].to(device, non_blocking=True)
                x_num = num_va[b_start: b_start + batch_size * 2].to(device, non_blocking=True)
                val_preds.append(model(x_cat, x_num).cpu().numpy())

        preds_arr = np.clip(np.concatenate(val_preds), config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        val_rmse_ep = rmse(yva_ap.to_numpy(), preds_arr)

        if val_rmse_ep < best_val_rmse:
            best_val_rmse = val_rmse_ep
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_preds = preds_arr
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_weights)
    return model, prep, best_val_rmse, best_preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--featureset", default=config.DEFAULT_FEATURESET,
                    choices=list(config.FEATURESETS))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
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

    Xtr, ytr, _ = D.load_part(args.split, "train", args.featureset,
                              filter_target=True, outlier_policy=args.outlier_policy)
    Xva, yva, meta_va = D.load_part(args.split, "val", args.featureset, filter_target=False)

    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]

    airports = sorted(Xtr[AIRPORT_COL].unique())
    print(f"[{ARM}] Training 10 dedicated airport models on {device} (split: {args.split})...")

    models = {}
    preps = {}
    airport_rmses = {}
    pred_va = np.zeros(len(yva), dtype="float64")

    t0 = time.time()
    for a in airports:
        m_tr = (Xtr[AIRPORT_COL] == a)
        m_va = (meta_va[AIRPORT_COL] == a)

        Xtr_ap = Xtr[m_tr].drop(columns=[AIRPORT_COL])
        ytr_ap = ytr[m_tr]
        Xva_ap = Xva[m_va].drop(columns=[AIRPORT_COL])
        yva_ap = yva[m_va]

        cat_ap = [c for c in cat_cols if c != AIRPORT_COL]

        t_start = time.time()
        model, prep, ap_rmse, ap_preds = train_single_airport(
            a, Xtr_ap, ytr_ap, Xva_ap, yva_ap, cat_ap, num_cols,
            device=device, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience, seed=args.seed
        )
        dur = time.time() - t_start
        print(f"  [{ARM}] Airport {a:6s} | Train N={len(Xtr_ap):6d} | Val N={len(Xva_ap):6d} | Val RMSE={ap_rmse:6.2f}s ({dur:.1f}s)")

        models[str(a)] = model.state_dict()
        preps[str(a)] = prep
        airport_rmses[str(a)] = float(ap_rmse)
        pred_va[m_va.to_numpy()] = ap_preds

    total_time = time.time() - t0
    overall_val_rmse = rmse(yva, pred_va)
    print(f"[{ARM}] Finished in {total_time:.1f}s. Overall Val RMSE: {overall_val_rmse:.4f} s")

    # Save models bundle and preprocessors
    torch.save(models, out / "models.pt")
    with open(out / "preprocessors.pkl", "wb") as fh:
        pickle.dump(preps, fh)

    model_meta = {
        "arm": ARM,
        "split": args.split,
        "featureset": args.featureset,
        "outlier_policy": args.outlier_policy,
        "seed": args.seed,
        "airport_rmses": airport_rmses,
        "val_rmse": float(overall_val_rmse),
        "train_pool_ids_sha": D.pool_fingerprint(Xtr.index),
        "feature_fingerprint": config.feature_code_fingerprint(),
    }
    write_json_guarded(out / "model.json", model_meta, force=args.force)

    res_val = E.evaluate(yva, pred_va, meta_va, ytr, part="val", arm=ARM, split=args.split,
                         featureset=args.featureset, seed=args.seed,
                         extra={"airport_rmses": airport_rmses})
    write_json_guarded(out / "eval_val.json", res_val, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
