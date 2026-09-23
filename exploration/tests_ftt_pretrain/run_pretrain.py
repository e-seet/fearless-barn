#!/usr/bin/env python3
"""Transductive masked-feature pretraining, then the baseline FTT fine-tune.

See PREREGISTERED.md -- the bar, the kill rule and the design were fixed there
before this file ran. Everything after pretraining is `tests_ftt/run_ftt.py`'s
forward recipe verbatim (optimizer, clipping, patience, early stopping on
forward val, prediction clipping, npz keys), so the saved predictions pair
seed-for-seed with `tests_ftt/preds_seed{s}_lr1e-3_e132.npz`.

Two stages per seed:

1. PRETRAIN on features only, from forward train + val + test + ranking. Mask
   30% of feature tokens, replace each with a learned per-feature mask
   embedding, and reconstruct: MSE on standardized numericals, CE on
   categoricals. Cells that were NaN before the median fill (and categoricals
   that mapped to "unknown") are excluded from the loss. Early-stop on a
   held-out 2% of corpus rows.
2. FINE-TUNE: re-seed, build a fresh FTTransformer exactly as the baseline
   does (so the RNG stream -- batch order, dropout -- matches), then overwrite
   its tokenizer + blocks with the pretrained ones. The regression head stays
   at the baseline's init.

usage:
  python tests_ftt_pretrain/run_pretrain.py 42 43 44 --kill-first-above 267.18
  python tests_ftt_pretrain/run_pretrain.py 42 --no-transductive --tag ctl  # control
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

FS, POL, SPLIT = "with_aobt", "keep_all", "forward"
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
STOP_FLAG = HERE / "STOP"


class MaskedPretrainer(nn.Module):
    """Wraps an FTTransformer's tokenizer + blocks with a mask embedding and
    per-feature reconstruction heads. Only `ftt.tokenizer` and `ftt.blocks` are
    carried into fine-tuning."""

    def __init__(self, ftt: FTTransformer, card: "dict[str, int]", n_num: int):
        super().__init__()
        self.ftt = ftt
        d = ftt.tokenizer.d_token
        self.cat_cols = list(card)
        self.n_cat, self.n_num = len(card), n_num
        n_feat = self.n_cat + n_num
        self.mask_emb = nn.Parameter(torch.empty(n_feat, d))
        nn.init.normal_(self.mask_emb, std=d ** -0.5)
        self.out_norm = nn.LayerNorm(d)
        self.cat_heads = nn.ModuleDict({c: nn.Linear(d, card[c] + 1) for c in card})
        self.num_w = nn.Parameter(torch.empty(n_num, d))
        self.num_b = nn.Parameter(torch.zeros(n_num))
        nn.init.normal_(self.num_w, std=d ** -0.5)

    def forward(self, x_cat, x_num, M):
        """M: (B, n_feat) bool, True = masked. Token order is cat then num,
        matching FeatureTokenizer."""
        tok = self.ftt.tokenizer(x_cat, x_num)                # (B, 1+F, d)
        feat = tok[:, 1:]
        feat = torch.where(M.unsqueeze(-1), self.mask_emb.unsqueeze(0), feat)
        x = torch.cat([tok[:, :1], feat], dim=1)
        for blk in self.ftt.blocks:
            x = blk(x)
        h = self.out_norm(x[:, 1:])
        cat_logits = [self.cat_heads[c](h[:, i]) for i, c in enumerate(self.cat_cols)]
        num_pred = (h[:, self.n_cat:] * self.num_w).sum(-1) + self.num_b
        return cat_logits, num_pred


def recon_loss(model, cat, num, valid, idx, mask_rate, gen):
    """Mean loss over masked AND valid cells. `valid`: (N, n_feat) bool."""
    B = len(idx)
    n_feat = valid.shape[1]
    M = torch.rand(B, n_feat, device=num.device, generator=gen) < mask_rate
    xc = {c: cat[c][idx] for c in cat}
    xn = num[idx]
    cat_logits, num_pred = model(xc, xn, M)
    use = M & valid[idx]
    tot, cnt = 0.0, use.sum()
    ce = nn.functional.cross_entropy
    for i, c in enumerate(model.cat_cols):
        u = use[:, i]
        if u.any():
            tot = tot + ce(cat_logits[i][u], xc[c][u], reduction="sum")
    un = use[:, model.n_cat:]
    tot = tot + ((num_pred - xn) ** 2)[un].sum()
    return tot / cnt.clamp(min=1), int(cnt)


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
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=175)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--pt-epochs", type=int, default=40)
    ap.add_argument("--pt-patience", type=int, default=3)
    ap.add_argument("--pt-batch-size", type=int, default=1024)
    ap.add_argument("--mask-rate", type=float, default=0.30)
    ap.add_argument("--no-transductive", action="store_true",
                    help="CONTROL: pretrain on forward train rows only")
    ap.add_argument("--kill-first-above", type=float, default=None,
                    help="pre-registered kill: stop after the FIRST seed if its "
                         "best val RMSE exceeds this")
    ap.add_argument("--tag", default="pt")
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Xtr, ytr, _ = D.load_part(SPLIT, "train", FS, filter_target=True, outlier_policy=POL)
    Xva, yva, _ = D.load_part(SPLIT, "val", FS, filter_target=False)
    Xte, yte, _ = D.load_part(SPLIT, "test", FS, filter_target=False)
    Xva, Xte = D.align_columns(Xtr, Xva), D.align_columns(Xtr, Xte)
    yva, yte = np.asarray(yva, float), np.asarray(yte, float)
    Xrk, _ = D.load_ranking_features(FS)
    Xrk = D.align_columns(Xtr, Xrk)

    assert len(Xtr) == len(ytr)
    _a, _y = np.asarray(Xtr["delta_aobt_sec"], float), np.asarray(ytr, float)
    _m = np.isfinite(_a) & np.isfinite(_y)
    _c = float(np.corrcoef(_a[_m], _y[_m])[0, 1])
    print(f"label-alignment corr = {_c:.3f} (known-good 0.54)", flush=True)
    assert _c > 0.3, "labels look misaligned"
    # pairing check against the baseline this is screened against
    base_meta = np.load(ROOT / "tests_ftt" / "meta.npz", allow_pickle=True)
    assert np.allclose(base_meta["yva"], yva, equal_nan=True), "val rows differ from baseline"
    assert np.allclose(base_meta["yte"], yte, equal_nan=True), "test rows differ from baseline"

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

    def enc(X):
        c, n = encode_data(X, cat_cols, num_cols, cat_maps, sc, med)
        return {k: v.to(dev) for k, v in c.items()}, n.to(dev)

    def valid_mask(X, c):
        vc = torch.stack([c[col] != len(cat_maps[col]) for col in cat_cols], dim=1)
        vn = torch.tensor(X[num_cols].notna().to_numpy(), device=dev)
        return torch.cat([vc, vn], dim=1)

    tr, va, te = enc(Xtr), enc(Xva), enc(Xte)
    corpus_X = [Xtr] if a.no_transductive else [Xtr, Xva, Xte, Xrk]
    parts = [enc(X) for X in corpus_X]
    pc = {c: torch.cat([p[0][c] for p in parts]) for c in cat_cols}
    pn = torch.cat([p[1] for p in parts])
    pvalid = torch.cat([valid_mask(X, p[0]) for X, p in zip(corpus_X, parts)])
    del parts
    N = len(pn)
    print(f"pretrain corpus: {N} rows ({'train only -- CONTROL' if a.no_transductive else 'train+val+test+ranking'}); "
          f"valid-cell rate {pvalid.float().mean():.4f}; "
          f"tokens {len(cat_cols)} cat + {len(num_cols)} num", flush=True)
    y_t = torch.tensor(np.asarray(ytr, dtype="float32"), device=dev)
    n = len(y_t)

    for si, seed in enumerate(a.seeds):
        out = HERE / f"preds_seed{seed}_{a.tag}.npz"
        if out.exists():
            print(f"seed {seed} exists, skip", flush=True)
            continue
        if STOP_FLAG.exists():
            print("STOP flag present -- exiting", flush=True)
            return 0
        t0 = time.time()

        # ---------------- stage 1: pretrain ----------------
        seed_everything(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        ftt = FTTransformer(card, num_features=len(num_cols),
                            prior_mean=float(np.mean(ytr))).to(dev)
        pt = MaskedPretrainer(ftt, card, len(num_cols)).to(dev)
        opt = torch.optim.AdamW(pt.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        g = torch.Generator(device=dev).manual_seed(seed)
        perm_all = torch.randperm(N, device=dev, generator=g)
        n_hold = N // 50
        hold, fit = perm_all[:n_hold], perm_all[n_hold:]
        g_hold = torch.Generator(device=dev)
        best_pt, best_pt_ep, bad, best_state, pt_hist = 1e18, 0, 0, None, []
        for ep in range(1, a.pt_epochs + 1):
            pt.train()
            order = fit[torch.randperm(len(fit), device=dev, generator=g)]
            run, k = 0.0, 0
            for i in range(0, len(order), a.pt_batch_size):
                opt.zero_grad()
                loss, _ = recon_loss(pt, pc, pn, pvalid, order[i:i + a.pt_batch_size],
                                     a.mask_rate, g)
                loss.backward()
                nn.utils.clip_grad_norm_(pt.parameters(), max_norm=5.0)
                opt.step()
                run += float(loss); k += 1
            pt.eval()
            g_hold.manual_seed(12345)                  # same masks every epoch
            with torch.no_grad():
                tot = cnt = 0.0
                for i in range(0, n_hold, 8192):
                    l, c = recon_loss(pt, pc, pn, pvalid, hold[i:i + 8192], a.mask_rate, g_hold)
                    tot += float(l) * c; cnt += c
            hl = tot / cnt
            pt_hist.append(hl)
            if hl < best_pt - 1e-4:
                best_pt, best_pt_ep, bad = hl, ep, 0
                best_state = {k2: v.detach().clone() for k2, v in pt.ftt.state_dict().items()}
            else:
                bad += 1
            print(f"  seed {seed} PRETRAIN ep {ep:2d}/{a.pt_epochs} train {run/k:.4f} "
                  f"hold {hl:.4f} (best {best_pt:.4f} @ {best_pt_ep}) [{time.time()-t0:.0f}s]",
                  flush=True)
            if bad >= a.pt_patience:
                break
        pt_hit_cap = ep >= a.pt_epochs and bad < a.pt_patience
        if pt_hit_cap:
            print(f"  seed {seed}: PRETRAIN HIT CAP -- unresolved, not failed", flush=True)
        t_pt = time.time() - t0
        del pt, ftt, opt
        torch.cuda.empty_cache()

        # ---------------- stage 2: fine-tune (baseline recipe) ----------------
        seed_everything(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = FTTransformer(card, num_features=len(num_cols),
                              prior_mean=float(np.mean(ytr))).to(dev)
        sd = model.state_dict()
        carried = {k2: v for k2, v in best_state.items()
                   if k2.startswith(("tokenizer.", "blocks."))}
        sd.update(carried)
        model.load_state_dict(sd)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        crit = nn.MSELoss()
        best, best_ep, bad, bpv, bpt = 1e18, 0, 0, None, None
        hist = []
        for ep in range(1, a.epochs + 1):
            model.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, a.batch_size):
                b = perm[i:i + a.batch_size]
                opt.zero_grad()
                loss = crit(model({c: tr[0][c][b] for c in tr[0]}, tr[1][b]), y_t[b])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
            pv = predict(model, *va)
            r = float(np.sqrt(np.mean((yva - pv) ** 2)))
            hist.append(r)
            if r < best:
                best, best_ep, bad = r, ep, 0
                bpv, bpt = pv, predict(model, *te)
            else:
                bad += 1
                if bad >= a.patience:
                    break
            print(f"  seed {seed} FINETUNE ep {ep:3d}/{a.epochs} val {r:.2f} "
                  f"(best {best:.2f} @ {best_ep}) [{time.time()-t0:.0f}s]", flush=True)
            if STOP_FLAG.exists():
                print("STOP flag present -- abandoning this seed unsaved", flush=True)
                return 0
        hit_cap = ep >= a.epochs and bad < a.patience
        torch.save({"state_dict": model.state_dict(), "cat_maps": cat_maps, "card": card,
                    "num_cols": num_cols, "cat_cols": cat_cols, "medians": med,
                    "scaler": sc}, HERE / f"model_seed{seed}_{a.tag}.pt")
        np.savez(out, best_epoch=best_ep, val=best, epochs_run=ep, hit_cap=hit_cap,
                 history=np.asarray(hist), pv=bpv, pt=bpt, lr=a.lr, n_params=n_par,
                 labels_aligned=True, split=SPLIT,
                 pt_best_epoch=best_pt_ep, pt_best_loss=best_pt, pt_hit_cap=pt_hit_cap,
                 pt_history=np.asarray(pt_hist), pt_seconds=t_pt,
                 transductive=not a.no_transductive, mask_rate=a.mask_rate)
        flag = "  *** HIT CAP -- truncated, do NOT screen ***" if hit_cap else ""
        print(f"seed {seed}: pretrain best @ {best_pt_ep} ({best_pt:.4f}); finetune "
              f"best_ep {best_ep} val {best:.2f} ran {ep}/{a.epochs} "
              f"[{time.time()-t0:.0f}s]{flag}", flush=True)
        if si == 0 and a.kill_first_above is not None and best > a.kill_first_above:
            print(f"PRE-REGISTERED KILL: seed {seed} best val {best:.2f} > "
                  f"{a.kill_first_above} -- stopping", flush=True)
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
