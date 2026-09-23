#!/usr/bin/env python3
"""Third-surface confirmation of the FTT lead on seasonal_full.

Bar fixed in advance: `PREREGISTERED_seasonal.md`. Protocol copied from
`tests_tabnet/confirm_seasonal_full.py`, which faced the same constraint:
seasonal_full is in config.SELECTION_FORBIDDEN_SPLITS, so **nothing is selected
here**. Per-airport weights for BOTH blends are solved on `forward_inner` and
applied frozen.

  2-member: tree + MLP6        (solve_alpha_per_airport, the shipped recipe)
  3-member: FTT + tree + MLP6  (per-airport simplex)

Six MLP seeds (42-47) because that is what exists as registered
`mlp_antigravity__seasonal_full` runs; the forward_inner solve is re-run at the
same six so the seed count matches between solve and apply.

Reported: seasonal_full TEST (Jan + Jul 2025) -- the honest read -- and Dec val,
which is FTT's own early-stopping surface and therefore NOT a bar.

usage: python tests_ftt/confirm_seasonal_full.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
from common.predict import predict_arm
from pipeline_airport_operator_multimodel.train import solve_alpha_per_airport
from pipeline_airport_operator_catboost_multimodel.train import (
    solve_weights_per_airport_simplex)

FS, POL = "with_aobt", "keep_all"
MLP_SEEDS = [42, 43, 44, 45, 46, 47]
FTT_SEEDS = [42, 43, 44]
SEED = 42
BAR_IN_BAND, BAR_NON_LIRF = 1.0, 1.0
CACHE = Path(os.environ.get(
    "FTT_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
    r"\2ff7d479-271f-463d-9593-1a05404d65d1\scratchpad\ftt_strong_cache"))
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def cached(name, fn):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{name}.npy"
    if f.exists():
        return np.load(f)
    v = fn()
    np.save(f, v)
    return v


def parts(split, part):
    Xtr, _, _ = D.load_part(split, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(split, part, FS, filter_target=False)
    X = D.align_columns(Xtr, X)
    y = np.asarray(y, float)
    apx = meta["ADEP_mvt"].astype(str).to_numpy()
    tree = cached(f"tree_{split}_{part}",
                  lambda: predict_arm("airport_operator_blend", split, part, FS, SEED,
                                      POL, X=X, meta=meta, Xtr=Xtr))
    mlp = cached(f"mlp{len(MLP_SEEDS)}_{split}_{part}",
                 lambda: np.mean([predict_arm("mlp_antigravity", split, part, FS, s,
                                              POL, X=X, meta=meta, Xtr=Xtr)
                                  for s in MLP_SEEDS], axis=0))
    return y, apx, tree, mlp


def ftt(split_sfx, key):
    ps = []
    for s in FTT_SEEDS:
        z = np.load(HERE / f"preds_seed{s}{split_sfx}.npz", allow_pickle=True)
        if bool(z["hit_cap"]):
            raise SystemExit(f"seed {s}{split_sfx} hit its epoch cap "
                             f"(best {int(z['best_epoch'])} of {int(z['epochs_run'])}) "
                             f"-- truncated, not converged. Raise --epochs and re-run.")
        ps.append(z[key])
    return np.mean(ps, axis=0)


def main() -> int:
    # ---- weights: solved on forward_inner, frozen ---------------------------
    yw, ap_w, tree_w, mlp_w = parts("forward_inner", "val")
    ftt_w = ftt("_forward_inner_lr1e-3", "pv")
    print(f"forward_inner val (weight solve, n={len(yw)}): "
          f"tree {rm(yw,tree_w):.2f} | mlp{len(MLP_SEEDS)} {rm(yw,mlp_w):.2f} | ftt {rm(yw,ftt_w):.2f}")
    alpha2 = solve_alpha_per_airport(yw, tree_w, mlp_w, ap_w)
    W3 = solve_weights_per_airport_simplex(yw, ftt_w, tree_w, mlp_w, ap_w)
    print(f"\nFTT weight at LIRF = {W3['LIRF'][0]:.3f} "
          f"(sanity: has been 0.000 in every prior solve)")
    print(f"{'airport':<8} {'2-mem tree/MLP':>18} | {'3-mem FTT/tree/MLP':>24}")
    for k in sorted(alpha2):
        w = W3[k]
        print(f"{k:<8} {alpha2[k]:7.3f}/{1-alpha2[k]:<7.3f}   | "
              f"{w[0]:7.3f}/{w[1]:7.3f}/{w[2]:7.3f}")

    out = {"mlp_seeds": MLP_SEEDS, "ftt_seeds": FTT_SEEDS, "bar": {
        "in_band": BAR_IN_BAND, "non_lirf": BAR_NON_LIRF},
        "alpha2": alpha2, "w3": W3}

    # ---- apply frozen on seasonal_full --------------------------------------
    for part, key, is_bar in (("test", "pt", True), ("val", "pv", False)):
        y, apx, tree, mlp = parts("seasonal_full", part)
        f = ftt("_seasonal_full_lr1e-3", key)
        assert len(f) == len(y), f"FTT {len(f)} vs seasonal_full {part} {len(y)}"
        two = np.zeros(len(y)); three = np.zeros(len(y))
        for k in np.unique(apx):
            m = apx == k
            al = alpha2.get(str(k), 0.5)
            two[m] = al * tree[m] + (1 - al) * mlp[m]
            w = W3.get(str(k), [1 / 3, 1 / 3, 1 / 3])
            three[m] = w[0] * f[m] + w[1] * tree[m] + w[2] * mlp[m]
        clip = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        two, three = np.clip(two, *clip), np.clip(three, *clip)

        L, ib = apx == "LIRF", y <= 7200
        label = "TEST (Jan + Jul)" if part == "test" else "Dec VAL (not a bar)"
        print(f"\n===== seasonal_full {label}  n={len(y)} =====")
        row = lambda nm, p: print(
            f"  {nm:22s} pooled {rm(y,p):8.2f} | in-band {rm(y[ib],p[ib]):7.2f} | "
            f"LIRF {rm(y[L],p[L]):9.2f} | non-LIRF {rm(y[~L],p[~L]):7.2f}")
        row(f"2-member (tree+MLP{len(MLP_SEEDS)})", two)
        row("3-member (+FTT)", three)
        d = {"pooled": rm(y, three) - rm(y, two),
             "in_band": rm(y[ib], three[ib]) - rm(y[ib], two[ib]),
             "non_lirf": rm(y[~L], three[~L]) - rm(y[~L], two[~L]),
             "lirf": rm(y[L], three[L]) - rm(y[L], two[L])}
        print(f"  FTT marginal: pooled {d['pooled']:+.2f} | in-band {d['in_band']:+.2f} | "
              f"LIRF {d['lirf']:+.2f} | non-LIRF {d['non_lirf']:+.2f}   (negative = helps)")
        out[part] = {"two": rm(y, two), "three": rm(y, three), "delta": d}
        if is_bar:
            ok = (-d["in_band"] >= BAR_IN_BAND) and (-d["non_lirf"] >= BAR_NON_LIRF)
            out["confirmed"] = bool(ok)
            print(f"\n  PRE-REGISTERED BAR (in-band gain >= {BAR_IN_BAND} AND "
                  f"non-LIRF gain >= {BAR_NON_LIRF}, both helping): "
                  f"{'CONFIRMED' if ok else 'NOT CONFIRMED'}")

    (HERE / "confirm_seasonal_full.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'confirm_seasonal_full.json'}")
    print("Reminder: NOT CONFIRMED means nothing is built and no slot is spent "
          "(PREREGISTERED_seasonal.md), and this check is not re-run at other settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
