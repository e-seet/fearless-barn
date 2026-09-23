#!/usr/bin/env python3
"""Does FTT still add once the MLP side is STRONG? The deflation check.

`solve_three_member.py` measured FTT against `airport_operator_multimodel`
tag=msmlp, whose MLP side is **3 seeds**. The shipped submission v25 averages
**18**. Seed-averaging is this build's single biggest measured lever (1->3->6->
12->18 seeds took the board 324 -> 306 -> 299 -> 295.75 -> 295.21), so a
3-seed MLP is a materially weaker partner than what FTT would actually have to
improve on, and -2.9 s is an upper bound rather than an estimate.

This runs the clean ablation. BOTH blends are solved from scratch on
`forward_inner` val with the SAME 12-seed MLP average (12 = every seed present
on both `forward` and `forward_inner`; forward_inner has 18 but forward only
has 42-53, and the evaluation needs both surfaces):

  2-member: tree + MLP12          -- solve_alpha_per_airport, the shipped recipe
  3-member: FTT + tree + MLP12    -- the per-airport simplex solver

so the ONLY difference between them is whether FTT is present. Both are then
frozen and applied to forward val (Sep-Oct) and test (Nov-Dec).

If the -2.9 s survives here it is a real addition to the shipped architecture.
If it collapses, FTT was substituting for MLP seeds rather than adding to them,
which is the `tests_matched_unmatched_router` failure in a subtler form.

Predictions are cached to the scratchpad so a re-run is cheap.

usage: python tests_ftt/solve_vs_strong_mlp.py [--mlp-seeds 42,...,53] [--l2 0.0]
"""
from __future__ import annotations

import argparse
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
FTT_SEEDS = [42, 43, 44]
SEED = 42
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


def surface(split, part, mlp_seeds):
    """(y, airports, tree, mlp_ens, in_band_mask) for one surface."""
    Xtr, _, _ = D.load_part(split, "train", FS, filter_target=True, outlier_policy=POL)
    X, y, meta = D.load_part(split, part, FS, filter_target=False)
    X = D.align_columns(Xtr, X)
    y = np.asarray(y, float)
    apx = meta["ADEP_mvt"].astype(str).to_numpy()
    tree = cached(f"tree_{split}_{part}",
                  lambda: predict_arm("airport_operator_blend", split, part, FS, SEED,
                                      POL, X=X, meta=meta, Xtr=Xtr))
    mlp = cached(f"mlp{len(mlp_seeds)}_{split}_{part}",
                 lambda: np.mean([predict_arm("mlp_antigravity", split, part, FS, s,
                                              POL, X=X, meta=meta, Xtr=Xtr)
                                  for s in mlp_seeds], axis=0))
    return y, apx, tree, mlp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mlp-seeds", default=",".join(str(s) for s in range(42, 54)))
    ap.add_argument("--l2", type=float, default=0.0)
    a = ap.parse_args(argv)
    mlp_seeds = [int(x) for x in a.mlp_seeds.split(",") if x.strip()]
    print(f"MLP seeds averaged: {mlp_seeds} (n={len(mlp_seeds)})")

    # ---- solve both blends on forward_inner val ----------------------------
    yw, ap_w, tree_w, mlp_w = surface("forward_inner", "val", mlp_seeds)
    ftt_w = np.mean([np.load(HERE / f"preds_seed{s}_forward_inner_lr1e-3.npz",
                             allow_pickle=True)["pv"] for s in FTT_SEEDS], axis=0)
    assert len(ftt_w) == len(yw)
    print(f"forward_inner val n={len(yw)}: tree {rm(yw,tree_w):.2f} | "
          f"mlp{len(mlp_seeds)} {rm(yw,mlp_w):.2f} | ftt {rm(yw,ftt_w):.2f}")

    # 2-member, the shipped recipe: alpha = weight on the TREE side
    alpha2 = solve_alpha_per_airport(yw, tree_w, mlp_w, ap_w)
    # 3-member simplex: [FTT, tree, MLP]
    W3 = solve_weights_per_airport_simplex(yw, ftt_w, tree_w, mlp_w, ap_w,
                                           l2_shrinkage=a.l2)
    print("\nper-airport weights")
    print(f"{'airport':<8} {'2-member tree/MLP':>20} | {'3-member FTT/tree/MLP':>26}")
    for k in sorted(alpha2):
        w = W3[k]
        print(f"{k:<8} {alpha2[k]:8.3f}/{1-alpha2[k]:<8.3f}    | "
              f"{w[0]:7.3f}/{w[1]:7.3f}/{w[2]:7.3f}")

    out = {"mlp_seeds": mlp_seeds, "ftt_seeds": FTT_SEEDS, "l2": a.l2,
           "alpha2_per_airport": alpha2, "w3_per_airport": W3}

    # ---- apply frozen to forward val and test ------------------------------
    for part in ("val", "test"):
        y, apx, tree, mlp = surface("forward", part, mlp_seeds)
        ftt = np.mean([np.load(HERE / f"preds_seed{s}_lr1e-3_e132.npz",
                               allow_pickle=True)["pv" if part == "val" else "pt"]
                       for s in FTT_SEEDS], axis=0)
        two = np.zeros(len(y))
        three = np.zeros(len(y))
        for k in np.unique(apx):
            m = apx == k
            al = alpha2.get(str(k), 0.5)
            two[m] = al * tree[m] + (1 - al) * mlp[m]
            w = W3.get(str(k), [1 / 3, 1 / 3, 1 / 3])
            three[m] = w[0] * ftt[m] + w[1] * tree[m] + w[2] * mlp[m]
        clip = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
        two, three = np.clip(two, *clip), np.clip(three, *clip)

        L, ib = apx == "LIRF", y <= 7200
        print(f"\n===== forward {part.upper()} (n={len(y)}) =====")
        row = lambda nm, p: print(
            f"  {nm:26s} pooled {rm(y,p):7.2f} | in-band {rm(y[ib],p[ib]):7.2f} | "
            f"LIRF {rm(y[L],p[L]):8.2f} | non-LIRF {rm(y[~L],p[~L]):7.2f}")
        row(f"2-member (tree+MLP{len(mlp_seeds)})", two)
        row("3-member (+FTT)", three)
        d = {"pooled": rm(y, three) - rm(y, two),
             "in_band": rm(y[ib], three[ib]) - rm(y[ib], two[ib]),
             "non_lirf": rm(y[~L], three[~L]) - rm(y[~L], two[~L]),
             "lirf": rm(y[L], three[L]) - rm(y[L], two[L])}
        print(f"  FTT's marginal contribution: pooled {d['pooled']:+.2f} | "
              f"in-band {d['in_band']:+.2f} | LIRF {d['lirf']:+.2f} | "
              f"non-LIRF {d['non_lirf']:+.2f}   (negative = FTT helps)")
        out[part] = {"two_member": rm(y, two), "three_member": rm(y, three), "delta": d}

    (HERE / "solve_vs_strong_mlp.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'solve_vs_strong_mlp.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
