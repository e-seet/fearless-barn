#!/usr/bin/env python3
"""Experiment: TabNet (sequential-attention deep tabular architecture) versus
this project's existing NN component (`mlp_antigravity`, entity-embedding
MLP) and the plain LightGBM control.

Why TabNet and not another deep-tabular option. Never tried in this repo --
the only NN so far is a plain entity-embedding MLP (`pipeline_mlp_antigravity`).
TabNet's inductive bias is genuinely different: sequential attention selects
a SUBSET of features at each of several decision steps (soft, learned feature
selection per step) rather than a dense MLP over the whole concatenated
embedding+numeric vector, or a GBM's greedy axis-aligned splits. If entity
embeddings + a dense backbone are already capturing what's learnable, this
should land close to the MLP; if per-step sparse feature attention captures
structure the dense MLP smooths over, it could do better -- an empirical
question, not assumed.

The correct comparator is `mlp_antigravity`, NOT plain LightGBM. Plain
LightGBM (val 297.78 / test 315.79, forward/with_aobt/keep_all seed42) is
already beaten by the existing MLP (val 253.99 / test 259.71, same protocol)
-- learned from the matched/unmatched router mistake (tests_matched_unmatched_
router/RESULT.md), where beating a weak baseline and beating the actually-
shipped component turned out to be two different questions.

Preprocessing mirrors pipeline_mlp_antigravity/train.py's encode_data (same
project convention): categoricals label-encoded from TRAIN vocabulary only
(unseen validation/test categories map to one shared "unknown" code, never a
leak), numeric columns median-filled (train medians) then standardized (train
scaler). Categorical embedding sizes handed to TabNet via cat_idxs/cat_dims/
cat_emb_dim, sized the same way as the existing MLP (`min(48, max(4, sqrt(card)))`)
for a fair architecture-only comparison, not a capacity-budget difference.

Single seed first (42), matching the project's cheap-screen-before-multi-seed
convention.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common import evaluate as E
from common import features as F
from common.bootstrap import run_dir, seed_everything

import os as _os
SPLIT, FEATURESET, POLICY = _os.environ.get("TABNET_SPLIT", "forward"), "with_aobt", "keep_all"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
CLIP_LO, CLIP_HI = config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC
import os
MAX_EPOCHS = int(os.environ.get("TABNET_MAX_EPOCHS", 100))
PATIENCE = int(os.environ.get("TABNET_PATIENCE", 12))
BATCH_SIZE = 4096
LR = float(os.environ.get("TABNET_LR", 2e-2))
TAG = os.environ.get("TABNET_TAG", "screen")


def clip(p):
    return np.clip(p, CLIP_LO, CLIP_HI)


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y, dtype="float64") - p) ** 2)))


def emb_dim(card):
    return min(48, max(4, int(card ** 0.5)))


def build_cat_maps(Xtr, cat_cols):
    maps = {}
    for c in cat_cols:
        cats = pd.unique(Xtr[c].astype("object").dropna())
        maps[c] = {v: i for i, v in enumerate(cats)}   # unknown code = len(cats)
    return maps


def encode(X, cat_cols, num_cols, cat_maps, medians, scaler):
    cat_arr = np.empty((len(X), len(cat_cols)), dtype="int64")
    for j, c in enumerate(cat_cols):
        m = cat_maps[c]
        unk = len(m)
        cat_arr[:, j] = X[c].astype("object").map(m).fillna(unk).to_numpy(dtype="int64")
    num_vals = X[num_cols].astype("float64").fillna(medians).to_numpy()
    num_arr = scaler.transform(num_vals).astype("float32")
    return cat_arr, num_arr


def main() -> int:
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor

    t0 = time.time()
    seed_everything(SEED)

    Xtr_raw, ytr, pool_meta = D.load_part(SPLIT, "train", FEATURESET,
                                          filter_target=True, outlier_policy=POLICY)
    Xva_raw, yva, meta_va = D.load_part(SPLIT, "val", FEATURESET, filter_target=False)
    Xte_raw, yte, meta_te = D.load_part(SPLIT, "test", FEATURESET, filter_target=False)
    Xva = D.align_columns(Xtr_raw, Xva_raw)
    Xte = D.align_columns(Xtr_raw, Xte_raw)
    ytr = np.asarray(ytr, dtype="float64")
    yva = np.asarray(yva, dtype="float64")
    yte = np.asarray(yte, dtype="float64")

    cat_cols = [c for c in F.CATEGORICAL if c in Xtr_raw.columns]
    num_cols = [c for c in Xtr_raw.columns if c not in cat_cols]
    print(f"loaded: train={len(Xtr_raw)} val={len(Xva)} test={len(Xte)}  "
          f"cats={len(cat_cols)} nums={len(num_cols)}  [{time.time()-t0:.0f}s]")

    cat_maps = build_cat_maps(Xtr_raw, cat_cols)
    cat_dims = [len(cat_maps[c]) + 1 for c in cat_cols]   # +1 for unknown code
    cat_emb_dims = [emb_dim(len(cat_maps[c])) for c in cat_cols]
    medians = Xtr_raw[num_cols].astype("float64").median()
    scaler = StandardScaler().fit(Xtr_raw[num_cols].astype("float64").fillna(medians).to_numpy())

    cat_tr, num_tr = encode(Xtr_raw, cat_cols, num_cols, cat_maps, medians, scaler)
    cat_va, num_va = encode(Xva, cat_cols, num_cols, cat_maps, medians, scaler)
    cat_te, num_te = encode(Xte, cat_cols, num_cols, cat_maps, medians, scaler)

    # TabNet takes one combined array; cat_idxs marks which columns are categorical
    Xtr_comb = np.concatenate([cat_tr, num_tr], axis=1)
    Xva_comb = np.concatenate([cat_va, num_va], axis=1)
    Xte_comb = np.concatenate([cat_te, num_te], axis=1)
    cat_idxs = list(range(len(cat_cols)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabNetRegressor(
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dims,
        n_d=64, n_a=64, n_steps=5, gamma=1.5, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.AdamW, optimizer_params={"lr": LR},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        seed=SEED, device_name=device, verbose=1,
    )
    print(f"training on device={device}  [{time.time()-t0:.0f}s]")
    model.fit(
        Xtr_comb, ytr.reshape(-1, 1),
        eval_set=[(Xva_comb, yva.reshape(-1, 1))], eval_name=["val"],
        eval_metric=["rmse"], max_epochs=MAX_EPOCHS, patience=PATIENCE,
        batch_size=BATCH_SIZE, virtual_batch_size=512, num_workers=0, drop_last=False,
    )
    n_epochs_run = len(model.history["loss"])
    hit_cap = n_epochs_run >= MAX_EPOCHS
    print(f"stopped after {n_epochs_run} epochs (cap {MAX_EPOCHS}, hit_cap={hit_cap})  "
          f"[{time.time()-t0:.0f}s]")

    p_val = clip(model.predict(Xva_comb).reshape(-1))
    p_test = clip(model.predict(Xte_comb).reshape(-1))

    np.savez(HERE / f"preds_seed{SEED}_{TAG}.npz", pv=p_val, pt=p_test, best_epoch=n_epochs_run)
    rv = E.evaluate(yva, p_val, meta_va, ytr, part="val", arm="tabnet",
                    split=SPLIT, featureset=FEATURESET, seed=SEED)
    rt = E.evaluate(yte, p_test, meta_te, ytr, part="test", arm="tabnet",
                    split=SPLIT, featureset=FEATURESET, seed=SEED)

    proto = config.protocol_key(SPLIT, FEATURESET, POLICY)
    d = run_dir(config.ARTIFACTS / "runs", "tabnet", protocol=proto, seed=SEED,
               extra={"tag": TAG})
    (d / "model.json").write_text(json.dumps({
        "arm": "tabnet", "split": SPLIT, "featureset": FEATURESET, "seed": SEED,
        "n_epochs_run": n_epochs_run, "hit_cap": hit_cap,
        "note": "tests_tabnet/run_tabnet.py -- diagnostic only",
    }, indent=2))
    (d / "eval_val.json").write_text(json.dumps(rv, indent=2))
    (d / "eval_test.json").write_text(json.dumps(rt, indent=2))

    MLP_VAL, MLP_TEST = 253.99, 259.71     # mlp_antigravity, same protocol, seed42
    LGB_VAL, LGB_TEST = 297.78, 315.79     # lightgbm l2ctl, same protocol, seed42
    lines = [
        "TabNet vs existing arms  (forward / with_aobt / keep_all, seed 42)",
        "=" * 70,
        f"{'arm':22s} {'val RMSE':>10s} {'test RMSE':>10s}",
        f"{'lightgbm (l2ctl)':22s} {LGB_VAL:10.2f} {LGB_TEST:10.2f}",
        f"{'mlp_antigravity':22s} {MLP_VAL:10.2f} {MLP_TEST:10.2f}",
        f"{'tabnet (this run)':22s} {rv['rmse']:10.2f} {rt['rmse']:10.2f}",
        "",
        f"tabnet vs mlp_antigravity: val {rv['rmse']-MLP_VAL:+.2f}  test {rt['rmse']-MLP_TEST:+.2f}",
        f"tabnet vs lightgbm:        val {rv['rmse']-LGB_VAL:+.2f}  test {rt['rmse']-LGB_TEST:+.2f}",
        f"val in-band {rv['rmse_in_band']:.2f}  test in-band {rt['rmse_in_band']:.2f}",
        f"epochs run {n_epochs_run}/{MAX_EPOCHS}  hit_cap={hit_cap}",
    ]
    txt = "\n".join(lines)
    (HERE / (f"summary_seed{SEED}" + ("" if TAG == "screen" else "_" + TAG) + ".txt")).write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print(f"\n[{time.time()-t0:.0f}s] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
