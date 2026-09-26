#!/usr/bin/env python3
"""TabNet for the LIRF-override build. Same recipe validated in run_tabnet.py
(lr 2e-3, n_d=n_a=64, n_steps=5, gamma 1.5, lambda_sparse 1e-4, AdamW, StepLR(10,0.9),
batch 4096, virtual 512, cap 60, patience 15, val-early-stopped best-epoch weights).

usage: python train_tabnet_pred.py {inner|final} SEED

inner : split forward_inner (train Jan-Jun, val Jul-Aug). Saves Jul-Aug val
        predictions + ids, used ONLY to solve the LIRF mix weight w against the
        MLP forward_inner arms (the project's weight-solve surface).
final : train Jan-Nov 2025, early-stop on Dec 2025 (a no-val refit is unsafe: TabNet
        best epoch varied 17-41 across validated runs). Saves Dec val preds and the
        predictions for every scored ranking DEP row (submitting.parquet order).
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common import features as F
from common.bootstrap import seed_everything

MODE, SEED = sys.argv[1], int(sys.argv[2])
FS, POL = "with_aobt", "keep_all"
# Cap raised 60 -> 75 on 2026-09-21. At 60, final seeds 47 and 51 ended AT the
# cap (best epochs 50 and 47) and the health gate excluded both. Re-run at 75
# they early-stopped at 66 and 63 with the SAME best epochs and byte-identical
# predictions: they were already at their optimum, the cap just stopped patience
# from confirming it. 75 = 1.5x the observed natural best of 50, the rule in
# tools/ml_common/convergence.py. See ../CONVERGENCE_AUDIT.md sec.3.
MAX_EPOCHS = int(os.environ.get("TABNET_MAX_EPOCHS", 75)); PATIENCE = 15; LR = 2e-3; BS = 4096
OUT = HERE / "build"; OUT.mkdir(exist_ok=True)
# TABNET_PREFIX (default ""): prefix for every output file. Set "rebuilt_" to retrain a
# recorded seed WITHOUT overwriting the npz that shipped submissions replay (their sha256
# is pinned in tabnet_lirf_override model.json).
PREFIX = os.environ.get("TABNET_PREFIX", "")
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
emb = lambda c: min(48, max(4, int(c ** 0.5)))


def main():
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor
    t0 = time.time(); seed_everything(SEED)
    if MODE == "inner":
        Xtr, ytr, _ = D.load_part("forward_inner", "train", FS, filter_target=True, outlier_policy=POL)
        Xva, yva, _ = D.load_part("forward_inner", "val", FS, filter_target=False)
        Xva = D.align_columns(Xtr, Xva); Xrk = None
    else:
        Xall = D.build_scope("train2025", FS).set_index(config.ID)
        y_all = D._targets().reindex(Xall.index)
        mo = Xall["month_num"].to_numpy()
        tr_m, va_m = mo <= 11, mo == 12
        # apply_outlier_policy returns (keep_mask, FULL-LENGTH y, info); like load_part, filter BOTH X and y with the mask.
        # (Bug fixed 2026-09-20: passing the unfiltered y against a filtered X shifted every label after the first dropped row.)
        keep, ytr_f, _ = D.apply_outlier_policy(y_all[tr_m], POL)
        kk = keep.to_numpy()
        Xtr = Xall[tr_m][kk]; ytr = np.asarray(ytr_f, float)[kk]
        Xva, yva = Xall[va_m], y_all[va_m]
        Xva = D.align_columns(Xtr, Xva)
        Xrk, meta_rk = D.load_ranking_features(FS); Xrk = D.align_columns(Xtr, Xrk)
    ytr = np.asarray(ytr, float); yva = np.asarray(yva, float)
    assert len(Xtr) == len(ytr), f"X/y length mismatch {len(Xtr)} vs {len(ytr)}"
    _a = np.asarray(Xtr["delta_aobt_sec"], float); _m = np.isfinite(_a) & np.isfinite(ytr)
    _c = float(np.corrcoef(_a[_m], ytr[_m])[0, 1]); print(f"[{MODE} seed {SEED}] label-alignment check corr(delta_aobt_sec, y) = {_c:.3f} (known-good pool 0.54)", flush=True)
    assert _c > 0.3, "labels look misaligned"
    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]; num_cols = [c for c in Xtr.columns if c not in cat_cols]
    maps = {c: {v: i for i, v in enumerate(pd.unique(Xtr[c].astype("object").dropna()))} for c in cat_cols}
    med = Xtr[num_cols].astype("float64").median()
    sc = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(med).to_numpy())

    def enc(X):
        ca = np.empty((len(X), len(cat_cols)), dtype="int64")
        for j, c in enumerate(cat_cols):
            ca[:, j] = X[c].astype("object").map(maps[c]).fillna(len(maps[c])).to_numpy(dtype="int64")
        na = sc.transform(X[num_cols].astype("float64").fillna(med).to_numpy()).astype("float32")
        return np.concatenate([ca, na], axis=1)

    Atr, Ava = enc(Xtr), enc(Xva)
    print(f"[{MODE} seed {SEED}] train {len(Atr)} val {len(Ava)} rank {0 if Xrk is None else len(Xrk)} [{time.time()-t0:.0f}s]", flush=True)
    m = TabNetRegressor(cat_idxs=list(range(len(cat_cols))), cat_dims=[len(maps[c]) + 1 for c in cat_cols],
                        cat_emb_dim=[emb(len(maps[c])) for c in cat_cols], n_d=64, n_a=64, n_steps=5, gamma=1.5,
                        lambda_sparse=1e-4, optimizer_fn=torch.optim.AdamW, optimizer_params={"lr": LR},
                        scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params={"step_size": 10, "gamma": 0.9},
                        seed=SEED, device_name="cuda" if torch.cuda.is_available() else "cpu", verbose=1)
    m.fit(Atr, ytr.reshape(-1, 1), eval_set=[(Ava, yva.reshape(-1, 1))], eval_name=["val"], eval_metric=["rmse"],
          max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=BS, virtual_batch_size=512, num_workers=0, drop_last=False)
    n_ep = len(m.history["loss"]); hit = n_ep >= MAX_EPOCHS
    pv = np.clip(m.predict(Ava).reshape(-1), *CLIP)
    save = dict(labels_aligned=True, pv=pv, val_ids=Xva.index.to_numpy(), yva=yva, best_epoch=int(m.best_epoch), epochs_run=n_ep, hit_cap=hit,
                val_rmse=float(np.sqrt(np.mean((yva - pv) ** 2))))
    if Xrk is not None:
        save["pr"] = np.clip(m.predict(enc(Xrk)).reshape(-1), *CLIP); save["rank_ids"] = Xrk.index.to_numpy()
    np.savez(OUT / f"{PREFIX}{MODE}_seed{SEED}.npz", **save)
    # Persist the MODEL, not just its predictions. The saved npz holds predictions
    # for the ranking rows that existed when it was written, so a refreshed
    # ranking.parquet strands them -- which is exactly why v29 carried
    # "not_reproducible_via_submit_py" (CONVERGENCE_AUDIT.md sec.6a). With the
    # binary on disk the recorded seeds can be re-predicted against a new scored
    # set instead of retrained. Seeds trained before this line have no binary.
    try:
        m.save_model(str(OUT / f"{PREFIX}{MODE}_seed{SEED}_model"))
        import pickle
        with open(OUT / f"{PREFIX}{MODE}_seed{SEED}_encoders.pkl", "wb") as fh:   # needed to encode NEW rows
            pickle.dump({"cat_cols": cat_cols, "num_cols": num_cols, "maps": maps, "medians": med, "scaler": sc}, fh)
        print(f"[{MODE} seed {SEED}] saved model + encoders -> {PREFIX}{MODE}_seed{SEED}_model.zip", flush=True)
    except Exception as e:                                    # never lose the preds over this
        print(f"[{MODE} seed {SEED}] WARNING: could not save model ({type(e).__name__}: {e})", flush=True)
    print(f"[{MODE} seed {SEED}] done val_rmse {save['val_rmse']:.2f} best_epoch {m.best_epoch} ran {n_ep}/{MAX_EPOCHS} hit_cap={hit} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
