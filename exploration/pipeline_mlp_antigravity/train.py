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
    ap.add_argument("--hidden-dims", default="256,128,64",
                    help="comma-separated MLP hidden layer sizes (default 256,128,64)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--residual", action="store_true",
                    help="use projected-skip-connection blocks instead of the "
                         "plain feed-forward stack (see model._ResidualBlock). "
                         "Off by default so existing behaviour is unchanged.")
    ap.add_argument("--lr-schedule", default="none", choices=["none", "cosine", "plateau"],
                    help="LR schedule over --epochs; 'none' keeps the flat AdamW "
                         "LR this arm has always used. 'plateau' needs val (not "
                         "available under --freeze-from refit).")
    ap.add_argument("--loss", default="mse", choices=["mse", "huber"],
                    help="training objective. 'huber' is quadratic below "
                         "--huber-delta and linear beyond it, softening the "
                         "gradient weight of large-residual rows without "
                         "excluding them outright (unlike --exclude-ids-file). "
                         "Model selection and reported RMSE are unaffected -- "
                         "both always use true squared error on val.")
    ap.add_argument("--huber-delta", type=float, default=150.0,
                    help="delta for --loss huber, in seconds. Only used when "
                         "--loss huber.")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--freeze-from", default=None)
    ap.add_argument("--outlier-policy", default=config.DEFAULT_OUTLIER_POLICY,
                    choices=sorted(config.OUTLIER_POLICIES))
    ap.add_argument("--congest", action="store_true",
                    help="attach the realised-taxi congestion columns "
                         "(pipeline_per_airport_congest/extra.py). Off by "
                         "default so this arm's behaviour is unchanged unless "
                         "asked for. Recorded in model.json AND in "
                         "preprocessor.pkl so evaluate and predict reproduce "
                         "the same feature frame.")
    ap.add_argument("--exclude-ids-file", default=None,
                    help="path to a JSON file containing a list of MVT_ID_mvt "
                         "values (or a dict of named lists -- pass the key via "
                         "--exclude-ids-key) to drop from the TRAINING pool "
                         "only, before fitting. Val/test are never filtered, "
                         "per this project's standing rule -- this only "
                         "removes rows the model learns FROM.")
    ap.add_argument("--exclude-ids-key", default=None,
                    help="key into --exclude-ids-file if it's a dict of named "
                         "lists rather than a bare list")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    _HIDDEN_DIMS = tuple(int(x) for x in str(args.hidden_dims).split(",") if x.strip())

    seed_everything(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    proto = config.protocol_key(args.split, args.featureset, args.outlier_policy)
    out = run_dir(config.ARTIFACTS / "runs", ARM, protocol=proto, seed=args.seed,
                  extra=({"tag": args.tag} if args.tag else None))

    Xtr, ytr, _pool_meta = D.load_part(args.split, "train", args.featureset,
                                       filter_target=True,
                                       outlier_policy=args.outlier_policy)
    n_excluded = 0
    if args.exclude_ids_file:
        raw_ids = json.loads(Path(args.exclude_ids_file).read_text())
        if isinstance(raw_ids, dict):
            raw_ids = raw_ids[args.exclude_ids_key]
        exclude_ids = pd.Index(raw_ids, dtype=Xtr.index.dtype)
        keep_mask = ~Xtr.index.isin(exclude_ids)
        n_excluded = int((~keep_mask).sum())
        Xtr = Xtr.loc[keep_mask]
        ytr = ytr.loc[keep_mask]
        print("[{}] excluded {} of {} requested training rows (--exclude-ids-file {}{})".format(
            ARM, n_excluded, len(exclude_ids), args.exclude_ids_file,
            " key={}".format(args.exclude_ids_key) if args.exclude_ids_key else ""))
    if args.congest:
        # Before cat_cols/num_cols are derived below -- those lists come from
        # Xtr.columns, and they are what encode_data indexes with at predict
        # time, so the columns have to be present here or they exist nowhere.
        from pipeline_per_airport_congest.extra import attach as attach_congest
        Xtr = attach_congest(Xtr, "train2025", args.featureset)
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
        if args.congest:
            # align_columns REINDEXES to Xtr's columns, so attaching only to the
            # train frame would manufacture the five columns as all-NaN on val,
            # which fillna(medians) then makes completely invisible.
            Xva = attach_congest(Xva, "train2025", args.featureset)
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
            "scaler": scaler, "medians": medians, "prior_mean": prior_mean,
            # Architecture, so evaluate/submit/ensemble rebuild the SAME shape
            # (a wider model's state_dict will not load into the default net).
            "hidden_dims": _HIDDEN_DIMS, "dropout": float(args.dropout),
            "residual": bool(args.residual),
            # Stamped into the bundle predict_base_arm already opens, so a
            # congest-trained model cannot be scored on a frame that lacks the
            # columns. (mirrors the "congest" key in airport_operator_blend's
            # model.pkl bundle.)
            "congest": bool(args.congest),
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

    model = TabularMLP(cat_cardinalities, num_features=len(num_cols),
                       hidden_dims=_HIDDEN_DIMS, dropout=float(args.dropout),
                       prior_mean=prior_mean, residual=bool(args.residual)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.loss == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta)
    else:
        criterion = nn.MSELoss()

    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif args.lr_schedule == "plateau":
        if refit_only:
            raise SystemExit("--lr-schedule plateau needs a val split; refit_only has none")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                                factor=0.5, patience=2)
    else:
        scheduler = None

    best_epoch = 0
    best_val_rmse = float("inf")
    patience_cnt = 0
    history = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        model.train()
        total_sq_err = 0.0
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

            # Always accumulate true squared error for the printed/reported
            # train RMSE, independent of --loss -- otherwise a Huber-trained
            # run's "train RMSE" log line would actually be a Huber-loss
            # pseudo-RMSE, not comparable across --loss values.
            total_sq_err += ((preds.detach() - y_b) ** 2).sum().item()
            n_samples += len(y_b)

        tr_rmse = np.sqrt(total_sq_err / n_samples)

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

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va_rmse)

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    sel = declare_selection_set("mlp_antigravity best_epoch",
                                selected_on="frozen" if refit_only else "val",
                                value=int(best_epoch))

    conv = convergence_evidence(history, max_epochs=args.epochs, patience=args.patience,
                                metric_key="val_rmse", mode="min") if not refit_only else {
        "refit_only": True, "epochs": best_epoch}

    write_json_guarded(out / "model.json", {
        "arm": ARM, "split": args.split, "featureset": args.featureset,
        "seed": args.seed, "best_epoch": int(best_epoch),
        "hidden_dims": list(_HIDDEN_DIMS), "dropout": float(args.dropout),
        "residual": bool(args.residual), "lr_schedule": args.lr_schedule,
        "loss": args.loss, "huber_delta": float(args.huber_delta) if args.loss == "huber" else None,
        "best_iteration": int(best_epoch),
        "congest": bool(args.congest),
        "exclude_ids_file": args.exclude_ids_file, "exclude_ids_key": args.exclude_ids_key,
        "n_excluded": n_excluded,
        "tag": args.tag,
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
