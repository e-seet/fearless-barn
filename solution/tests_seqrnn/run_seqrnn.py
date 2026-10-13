#!/usr/bin/env python3
"""Experiment: a GRU sequence-state encoder over each departure's backward-only
movement history (DEP+ARR, `sequence_features.py`), joined to the same
entity-embedding tabular MLP architecture as `mlp_antigravity`, versus that
arm unchanged.

This is the genuinely-sequential construction the project's own notes
flagged as untested: `pipeline_queueattn_antigravity` already closed flat at
6 seeds (mean diff +0.670s worse, not distinguishable from zero) but that
architecture is fixed-K bidirectional ATTENTION over other departures only --
not causal, not recurrent, no arrivals. This tests whether a true causal
recurrent state (GRU over the last K=24 movements of BOTH phases, oldest to
newest) captures something the attention-pool construction doesn't.

Correct comparator: `mlp_antigravity` (val 253.99 / test 259.71, same
protocol, seed 42) -- the tabular half of this model is byte-identical in
shape to that arm, so any difference is attributable to the added GRU
branch, not a capacity or preprocessing confound. Single seed first (42),
matching this project's cheap-screen convention -- confirmed necessary by
the TabNet screen (tests_tabnet/RESULT.md), where seed 42 alone was a
misleading outlier.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import evaluate as E
from common import features as F
from common.bootstrap import run_dir, seed_everything
from tests_seqrnn.model import SeqTabularModel
from tests_seqrnn import sequence_features as SF

SPLIT, FEATURESET, POLICY = "forward", "with_aobt", "keep_all"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC
EPOCHS, BATCH_SIZE, LR, PATIENCE = 25, 2048, 1e-3, 6
HIDDEN_DIMS, DROPOUT, GRU_HIDDEN = (256, 128, 64), 0.1, 32


class SeqTabularDataset(Dataset):
    def __init__(self, cat_tensors, num_tensor, seq_tensor, mask_tensor, targets=None):
        self.cat_tensors = cat_tensors
        self.num_tensor = num_tensor
        self.seq_tensor = seq_tensor
        self.mask_tensor = mask_tensor
        self.targets = targets

    def __len__(self):
        return len(self.num_tensor)

    def __getitem__(self, idx):
        x_cat = {c: self.cat_tensors[c][idx] for c in self.cat_tensors}
        item = (x_cat, self.num_tensor[idx], self.seq_tensor[idx], self.mask_tensor[idx])
        if self.targets is not None:
            return item + (self.targets[idx],)
        return item


def encode_data(X, cat_cols, num_cols, cat_maps, scaler, medians):
    cat_tensors = {}
    for col in cat_cols:
        m = cat_maps[col]
        unk = len(m)
        vals = X[col].astype("object").map(m).fillna(unk).to_numpy(dtype="int64")
        cat_tensors[col] = torch.tensor(vals, dtype=torch.long)
    num_vals = X[num_cols].astype("float64").fillna(medians).to_numpy()
    num_scaled = scaler.transform(num_vals)
    return cat_tensors, torch.tensor(num_scaled, dtype=torch.float32)


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype="float64") - p) ** 2)))


def main() -> int:
    t0 = time.time()
    seed_everything(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, pool_meta = D.load_part(SPLIT, "train", FEATURESET,
                                      filter_target=True, outlier_policy=POLICY)
    Xva, yva, meta_va = D.load_part(SPLIT, "val", FEATURESET, filter_target=False)
    Xte, yte, meta_te = D.load_part(SPLIT, "test", FEATURESET, filter_target=False)
    Xva = D.align_columns(Xtr, Xva)
    Xte = D.align_columns(Xtr, Xte)
    ytr = np.asarray(ytr, dtype="float64")
    yva = np.asarray(yva, dtype="float64")
    yte = np.asarray(yte, dtype="float64")

    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]
    print(f"loaded tabular: train={len(Xtr)} val={len(Xva)} test={len(Xte)}  "
          f"cats={len(cat_cols)} nums={len(num_cols)}  [{time.time()-t0:.0f}s]")

    # ---- sequence tensors (cached, keyed on this file's own fingerprint) ----
    seq_train2025 = SF.load_scope_cached("train2025")
    seq_tr = SF.align_to_ids(seq_train2025, Xtr.index.to_numpy())
    seq_va = SF.align_to_ids(seq_train2025, Xva.index.to_numpy())
    seq_te = SF.align_to_ids(seq_train2025, Xte.index.to_numpy())
    print(f"sequence tensors built/loaded, K={SF.K}  "
          f"train coverage={seq_tr['mask'].any(axis=1).mean():.3%}  "
          f"[{time.time()-t0:.0f}s]")

    cat_maps, cat_cardinalities = {}, {}
    for col in cat_cols:
        cats = sorted(Xtr[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        cat_cardinalities[col] = len(cats)
    medians = Xtr[num_cols].astype("float64").median()
    scaler = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(medians).to_numpy())
    prior_mean = float(ytr.mean())

    tr_cat, tr_num = encode_data(Xtr, cat_cols, num_cols, cat_maps, scaler, medians)
    va_cat, va_num = encode_data(Xva, cat_cols, num_cols, cat_maps, scaler, medians)
    te_cat, te_num = encode_data(Xte, cat_cols, num_cols, cat_maps, scaler, medians)

    tr_seq = torch.tensor(seq_tr["seq"], dtype=torch.float32)
    tr_mask = torch.tensor(seq_tr["mask"], dtype=torch.float32)
    va_seq = torch.tensor(seq_va["seq"], dtype=torch.float32)
    va_mask = torch.tensor(seq_va["mask"], dtype=torch.float32)
    te_seq = torch.tensor(seq_te["seq"], dtype=torch.float32)
    te_mask = torch.tensor(seq_te["mask"], dtype=torch.float32)

    tr_targets = torch.tensor(ytr.astype("float32"))
    tr_ds = SeqTabularDataset(tr_cat, tr_num, tr_seq, tr_mask, tr_targets)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=0)
    va_ds = SeqTabularDataset(va_cat, va_num, va_seq, va_mask)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)

    model = SeqTabularModel(cat_cardinalities, num_features=len(num_cols),
                            seq_feature_dim=SF.SEQ_FEATURE_DIM, gru_hidden=GRU_HIDDEN,
                            hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
                            prior_mean=prior_mean).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.MSELoss()

    proto = config.protocol_key(SPLIT, FEATURESET, POLICY)
    out = run_dir(config.ARTIFACTS / "runs", "seqrnn", protocol=proto, seed=SEED,
                 extra={"tag": "screen"})

    best_epoch, best_val_rmse, patience_cnt = 0, float("inf"), 0
    for epoch in range(1, EPOCHS + 1):
        te0 = time.time()
        model.train()
        sse, n = 0.0, 0
        for x_cat, x_num, x_seq, x_mask, y_b in tr_loader:
            x_cat = {c: x_cat[c].to(device) for c in x_cat}
            x_num, x_seq, x_mask, y_b = (x_num.to(device), x_seq.to(device),
                                         x_mask.to(device), y_b.to(device))
            optimizer.zero_grad()
            preds = model(x_cat, x_num, x_seq, x_mask)
            loss = criterion(preds, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            sse += ((preds.detach() - y_b) ** 2).sum().item()
            n += len(y_b)
        tr_rmse = np.sqrt(sse / n)

        model.eval()
        va_preds = []
        with torch.no_grad():
            for x_cat, x_num, x_seq, x_mask in va_loader:
                x_cat = {c: x_cat[c].to(device) for c in x_cat}
                x_num, x_seq, x_mask = x_num.to(device), x_seq.to(device), x_mask.to(device)
                va_preds.append(model(x_cat, x_num, x_seq, x_mask).cpu().numpy())
        va_preds = clip(np.concatenate(va_preds))
        va_rmse = rmse(yva, va_preds)
        print(f"  epoch {epoch:2d}/{EPOCHS} [{time.time()-te0:.1f}s] "
              f"train RMSE={tr_rmse:.3f} val RMSE={va_rmse:.3f}")

        if va_rmse < best_val_rmse:
            best_val_rmse, best_epoch, patience_cnt = va_rmse, epoch, 0
            torch.save(model.state_dict(), out / "model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  early stopping fired after epoch {epoch} (best epoch {best_epoch})")
                break

    model.load_state_dict(torch.load(out / "model.pt"))
    model.eval()

    def predict(cat, num, seq, mask, n_rows):
        ds = SeqTabularDataset(cat, num, seq, mask)
        loader = DataLoader(ds, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0)
        preds = []
        with torch.no_grad():
            for x_cat, x_num, x_seq, x_mask in loader:
                x_cat = {c: x_cat[c].to(device) for c in x_cat}
                x_num, x_seq, x_mask = x_num.to(device), x_seq.to(device), x_mask.to(device)
                preds.append(model(x_cat, x_num, x_seq, x_mask).cpu().numpy())
        return clip(np.concatenate(preds))

    p_val = predict(va_cat, va_num, va_seq, va_mask, len(yva))
    p_test = predict(te_cat, te_num, te_seq, te_mask, len(yte))

    rv = E.evaluate(yva, p_val, meta_va, ytr, part="val", arm="seqrnn",
                    split=SPLIT, featureset=FEATURESET, seed=SEED)
    rt = E.evaluate(yte, p_test, meta_te, ytr, part="test", arm="seqrnn",
                    split=SPLIT, featureset=FEATURESET, seed=SEED)
    (out / "model.json").write_text(json.dumps({
        "arm": "seqrnn", "split": SPLIT, "featureset": FEATURESET, "seed": SEED,
        "best_epoch": best_epoch, "k": SF.K, "gru_hidden": GRU_HIDDEN,
        "note": "tests_seqrnn/run_seqrnn.py -- diagnostic only",
    }, indent=2))
    (out / "eval_val.json").write_text(json.dumps(rv, indent=2))
    (out / "eval_test.json").write_text(json.dumps(rt, indent=2))

    MLP_VAL, MLP_TEST = 253.99, 259.71
    LGB_VAL, LGB_TEST = 297.78, 315.79
    lines = [
        f"seqrnn (GRU state) vs existing arms  (forward / with_aobt / keep_all, seed {SEED})",
        "=" * 74,
        f"{'arm':22s} {'val RMSE':>10s} {'test RMSE':>10s}",
        f"{'lightgbm (l2ctl)':22s} {LGB_VAL:10.2f} {LGB_TEST:10.2f}",
        f"{'mlp_antigravity':22s} {MLP_VAL:10.2f} {MLP_TEST:10.2f}",
        f"{'seqrnn (this run)':22s} {rv['rmse']:10.2f} {rt['rmse']:10.2f}",
        "",
        f"seqrnn vs mlp_antigravity: val {rv['rmse']-MLP_VAL:+.2f}  test {rt['rmse']-MLP_TEST:+.2f}",
        f"seqrnn vs lightgbm:        val {rv['rmse']-LGB_VAL:+.2f}  test {rt['rmse']-LGB_TEST:+.2f}",
        f"val in-band {rv['rmse_in_band']:.2f}  test in-band {rt['rmse_in_band']:.2f}",
        f"best_epoch {best_epoch}/{EPOCHS}",
    ]
    txt = "\n".join(lines)
    (HERE / f"summary_seed{SEED}.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n[{time.time()-t0:.0f}s] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
