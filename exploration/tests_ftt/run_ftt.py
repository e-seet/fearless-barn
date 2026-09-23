#!/usr/bin/env python3
"""Train FT-Transformer seeds on forward/with_aobt/keep_all and save val/test
predictions in the SAME npz format as tests_lirf_seeds/train_seeds.py.

Everything except the architecture is copied from that script on purpose -- same
features, same split, same AdamW/clip/early-stopping recipe, same prediction
clipping, same output keys -- so the saved preds drop straight into the existing
decorrelation screen and any delta is attributable to the model.

Two deliberate deviations, both recorded before the first run:

* **lr defaults to 1e-4, not the MLP's 1e-3.** Transformers on tabular data are
  lr-sensitive and this build already paid for that lesson once: TabNet's
  original "decisive loss" verdict at lr 2e-2 turned out to be an
  optimiser-instability artefact and had to be retracted after a rerun at 2e-3
  (tests_tabnet/RESULT.md). 1e-4 with weight decay 1e-5 is the FT-Transformer
  paper's own default. `--lr` is exposed so the sensitivity can be checked
  rather than assumed.
* **--epochs defaults to 60, patience 10.** There is no observed best epoch for
  this architecture yet, so the cap cannot be set by the repo's
  1.5x-observed-best rule (tools/ml_common/convergence.py note) until the first
  run produces one. The cap is generous on purpose and `hit_cap` is saved in
  every npz: a run that ends AT the cap is truncated, not converged, and must
  not be screened (../CONVERGENCE_AUDIT.md sec.1/sec.3).

usage:
  python tests_ftt/run_ftt.py 42 43 44 --lr 1e-3 --epochs 132 --tag lr1e-3_e132
  python tests_ftt/run_ftt.py 42 43 44 --split forward_inner --lr 1e-3       --epochs 180 --tag lr1e-3

--split forward_inner trains on Jan-Jun and predicts Jul-Aug: that val surface
is where solve_alpha_per_airport fits the per-airport blend weights, so an
honest FTT weight has to come from there rather than from forward's val (which
the screen already reports on, and which would be selecting on the surface being
scored). forward_inner has no test part, so only `pv` is saved.
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
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common import features as F
from common.bootstrap import seed_everything
from pipeline_mlp_antigravity.train import encode_data
from tests_ftt.model import FTTransformer

FS, POL = "with_aobt", "keep_all"
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


@torch.no_grad()
def predict(model, cat, num, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(num), bs):
        out.append(model({c: cat[c][i:i + bs] for c in cat},
                         num[i:i + bs]).float().cpu().numpy())
    return np.clip(np.concatenate(out), *CLIP)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--d-token", type=int, default=64)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--freeze-epochs", type=int, default=None,
                    help="REQUIRED for a split with no val (e.g. `full`): train "
                         "for exactly this many epochs with no early stopping. "
                         "Mirrors mlp_antigravity's --freeze-from refit. Per "
                         "../CONVERGENCE_AUDIT.md sec.2, pass the MEDIAN best "
                         "epoch across the forward seeds rather than one seed's, "
                         "and use the same value for every member.")
    ap.add_argument("--predict-ranking", action="store_true",
                    help="also predict the scored ranking rows and save `pr` + "
                         "`rank_ids` (needed to build a submission).")
    ap.add_argument("--split", default="forward", choices=sorted(config.SPLITS),
                    help="forward_inner is the weight-solve surface: its val "
                         "(Jul-Aug) is where solve_alpha_per_airport fits the "
                         "per-airport blend weights, so an honest weight needs "
                         "FTT predictions there, not on forward's val.")
    a = ap.parse_args(argv)
    suffix = f"_{a.tag}" if a.tag else ""

    SPLIT = a.split
    has_test = bool(config.SPLITS[SPLIT]["test"])
    refit_only = not config.SPLITS[SPLIT]["val"]
    if refit_only and not a.freeze_epochs:
        raise SystemExit(f"split '{SPLIT}' has no val -- pass --freeze-epochs N "
                         f"(see --help; N should be the MEDIAN forward best epoch).")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, ytr, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    if refit_only:
        Xva = yva = mva = None
    else:
        Xva, yva, mva = D.load_part(SPLIT, "val", FS, filter_target=False)
        Xva = D.align_columns(Xtr, Xva)
        yva = np.asarray(yva, float)
    if has_test:
        Xte, yte, mte = D.load_part(SPLIT, "test", FS, filter_target=False)
        Xte = D.align_columns(Xtr, Xte)
        yte = np.asarray(yte, float)
    else:
        # forward_inner has no test part -- it exists to be a weight-solve
        # holdout, not an evaluation surface.
        Xte = yte = mte = None

    # The same label-alignment assert that caught the X/y shift in the TabNet
    # build (unfiltered y against a filtered X). Cheap, and it turns a silent
    # 'this architecture failed' into a visible bug.
    assert len(Xtr) == len(ytr), f"X/y length mismatch {len(Xtr)} vs {len(ytr)}"
    _a = np.asarray(Xtr["delta_aobt_sec"], float)
    _y = np.asarray(ytr, float)
    _m = np.isfinite(_a) & np.isfinite(_y)
    _c = float(np.corrcoef(_a[_m], _y[_m])[0, 1])
    print(f"label-alignment check corr(delta_aobt_sec, y) = {_c:.3f} (known-good 0.54)", flush=True)
    assert _c > 0.3, "labels look misaligned"

    cat_cols = [c for c in F.CATEGORICAL if c in Xtr.columns]
    num_cols = [c for c in Xtr.columns
                if c not in cat_cols and pd.api.types.is_numeric_dtype(Xtr[c])]
    cat_maps, card = {}, {}
    for col in cat_cols:
        cats = sorted(Xtr[col].dropna().astype("object").unique())
        cat_maps[col] = {v: i for i, v in enumerate(cats)}
        card[col] = len(cats)
    med = Xtr[num_cols].astype("float64").median()
    sc = StandardScaler().fit(Xtr[num_cols].astype("float64").fillna(med).to_numpy())
    todev = lambda cn: ({c: t.to(dev) for c, t in cn[0].items()}, cn[1].to(dev))
    tr = todev(encode_data(Xtr, cat_cols, num_cols, cat_maps, sc, med))
    va = todev(encode_data(Xva, cat_cols, num_cols, cat_maps, sc, med)) if not refit_only else None
    if a.predict_ranking:
        Xrk, meta_rk = D.load_ranking_features(FS)
        Xrk = D.align_columns(Xtr, Xrk)
        rk = todev(encode_data(Xrk, cat_cols, num_cols, cat_maps, sc, med))
        print(f"ranking rows: {len(Xrk)}", flush=True)
    te = todev(encode_data(Xte, cat_cols, num_cols, cat_maps, sc, med)) if has_test else None
    y_t = torch.tensor(np.asarray(ytr, dtype="float32"), device=dev)
    n = len(y_t)
    print(f"tokens = {len(cat_cols)} cat + {len(num_cols)} num + 1 CLS = "
          f"{len(cat_cols) + len(num_cols) + 1}; train n={n}", flush=True)

    # meta.npz is identical in content to tests_lirf_seeds/meta.npz (same split,
    # same rows); written here too so this folder stands alone.
    meta_name = "meta.npz" if SPLIT == "forward" else f"meta_{SPLIT}.npz"
    meta_kw = {} if refit_only else dict(
        yva=yva, apva=mva["ADEP_mvt"].astype(str).to_numpy(),
        val_ids=Xva.index.to_numpy())
    if has_test:
        meta_kw.update(yte=yte, apte=mte["ADEP_mvt"].astype(str).to_numpy(),
                       test_ids=Xte.index.to_numpy())
    np.savez(HERE / meta_name, **meta_kw)

    for seed in a.seeds:
        sp = "" if SPLIT == "forward" else f"_{SPLIT}"
        out = HERE / f"preds_seed{seed}{sp}{suffix}.npz"
        if out.exists():
            print(f"seed {seed} exists, skip", flush=True)
            continue
        t0 = time.time()
        seed_everything(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = FTTransformer(card, num_features=len(num_cols), d_token=a.d_token,
                              n_blocks=a.n_blocks, n_heads=a.n_heads,
                              prior_mean=float(np.mean(ytr))).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        crit = nn.MSELoss()
        best, best_ep, bad, bpv, bpt = 1e18, 0, 0, None, None
        hist = []
        n_epochs = a.freeze_epochs if refit_only else a.epochs
        for ep in range(1, n_epochs + 1):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, a.batch_size):
                b = perm[i:i + a.batch_size]
                opt.zero_grad()
                loss = crit(model({c: tr[0][c][b] for c in tr[0]}, tr[1][b]), y_t[b])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
            if refit_only:
                # No val by construction: train the frozen number of epochs and
                # keep the LAST one. There is nothing to select on, which is the
                # whole reason `full` is in SELECTION_FORBIDDEN_SPLITS.
                best_ep = ep
                print(f"  seed {seed} epoch {ep:3d}/{n_epochs} (refit, no val) "
                      f"[{time.time()-t0:.0f}s]", flush=True)
                continue
            pv = predict(model, *va)
            r = float(np.sqrt(np.mean((yva - pv) ** 2)))
            hist.append(r)
            if r < best:
                best, best_ep, bad = r, ep, 0
                bpv = pv
                bpt = predict(model, *te) if has_test else None
            else:
                bad += 1
                if bad >= a.patience:
                    break
            print(f"  seed {seed} epoch {ep:2d}/{a.epochs} val {r:.2f} "
                  f"(best {best:.2f} @ {best_ep}) [{time.time()-t0:.0f}s]", flush=True)
        hit_cap = (not refit_only) and ep >= a.epochs and bad < a.patience
        save_kw = dict(best_epoch=best_ep, val=best, epochs_run=ep,
                       hit_cap=hit_cap, history=np.asarray(hist), lr=a.lr,
                       n_params=n_par, labels_aligned=True, split=SPLIT,
                       refit_only=refit_only)
        if not refit_only:
            save_kw.update(pv=bpv, val_ids=Xva.index.to_numpy())
        if has_test:
            save_kw["pt"] = bpt
        if a.predict_ranking:
            save_kw["pr"] = predict(model, *rk)
            save_kw["rank_ids"] = Xrk.index.to_numpy()
        # Persist the model, not only its predictions -- the gap that made v29
        # unrebuildable (../CONVERGENCE_AUDIT.md sec.6a).
        torch.save({"state_dict": model.state_dict(), "cat_maps": cat_maps,
                    "card": card, "num_cols": num_cols, "cat_cols": cat_cols,
                    "medians": med, "scaler": sc, "d_token": a.d_token,
                    "n_blocks": a.n_blocks, "n_heads": a.n_heads},
                   HERE / f"model_seed{seed}{sp}{suffix}.pt")
        np.savez(out, **save_kw)
        flag = "  *** HIT CAP -- truncated, do NOT screen this seed ***" if hit_cap else ""
        print(f"seed {seed}: best_ep {best_ep} val {best:.2f} ran {ep}/{a.epochs} "
              f"params {n_par/1e6:.2f}M [{time.time()-t0:.0f}s]{flag}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
