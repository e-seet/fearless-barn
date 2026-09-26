#!/usr/bin/env python3
"""Build the season-weighting board probe (PROBE.md). Deterministic.

    python tests_season_weight/build_probe.py                 # scratch check only
    python tests_season_weight/build_probe.py --version 37    # also write submissions/ (user go-ahead)

Not a registered submit.py arm; the sidecar meta says so. If the probe wins on the
board it gets registered before any further use.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
import submit
from common import dataset as D
from pipeline_airport_operator_catboost_multimodel.train import solve_weights_per_airport_simplex
from pipeline_ftt_multimodel.train import blend_per_airport
from pipeline_tabnet_lirf_override.train import apply_override, seed_preds_path
from tests_ftt.solve_final_weights import CACHE

SEEDS, FTT6, TABNET, TW = list(range(42, 60)), list(range(42, 48)), [42, 43, 45, 46, 48], 0.459
F = ROOT / "tests_ftt"
rms = lambda a, b: float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def season_mlp(split, targets, months, key):
    out = np.full(len(months), np.nan)
    for t in targets:
        zs = [np.load(HERE / f"preds_{split}_t{t:02d}_seed{s}.npz", allow_pickle=True) for s in SEEDS]
        if any(bool(z["hit_cap"]) for z in zs):
            raise SystemExit(f"{split} t{t:02d}: a seed hit its cap")
        m = months == t
        out[m] = np.mean([z[key] for z in zs], axis=0)[m]
    assert np.isfinite(out).all(), f"{split}: rows with no target-month model"
    return out, zs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=None)
    a = ap.parse_args()

    # ---- weights, solved on forward_inner exactly as v35's (default solver) ----
    mi = np.load(HERE / "meta_forward_inner.npz", allow_pickle=True)
    yw, apw, mw_month = mi["yva"], mi["apva"], mi["mva"]
    tree_w = np.load(CACHE / "tree_forward_inner_val.npy")
    ftt_w = np.mean([np.load(F / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in FTT6], axis=0)
    mlp_w, _ = season_mlp("forward_inner", [7, 8], mw_month, "pv")
    assert len(yw) == len(tree_w) == len(ftt_w) == len(mlp_w)
    W = solve_weights_per_airport_simplex(yw, ftt_w, tree_w, mlp_w, apw, l2_shrinkage=0.0)
    if W["LIRF"][0] > 1e-6:
        raise SystemExit(f"HALT: FTT weight at LIRF = {W['LIRF'][0]}")
    print("weights [FTT, tree, MLP]:", {k: [round(x, 3) for x in v] for k, v in sorted(W.items())})

    # ---- scored set ----
    ids = pd.read_parquet(config.SUBMITTING_PARQUET)[config.ID].to_numpy()
    tree, meta = submit.predict_ranking("airport_operator_blend", "full", "with_aobt", 42, "keep_all")
    rank_ids = meta.index.to_numpy()
    assert np.array_equal(rank_ids, ids), "ranking order differs from the submission template"
    ap_r = meta["ADEP_mvt"].astype(str).to_numpy()
    ftt = np.mean([np.load(F / f"preds_seed{s}_full_lr1e-3.npz", allow_pickle=True)["pr"] for s in FTT6], axis=0)
    z0 = np.load(HERE / f"preds_full_t01_seed{SEEDS[0]}.npz", allow_pickle=True)
    assert np.array_equal(z0["rank_ids"], rank_ids), "season MLP ranking rows differ"
    mlp, _ = season_mlp("full", [1, 7], z0["rank_month"], "pr")
    base = blend_per_airport(ftt, tree, mlp, ap_r, W)
    tab = np.mean([np.load(seed_preds_path(s), allow_pickle=True)["pr"] for s in TABNET], axis=0)
    pred, n_over = apply_override(base, tab, ap_r, TW)
    sub = pd.DataFrame({config.ID: ids, config.TARGET: np.rint(pred).astype("int32")})
    errs = submit.validate(sub)

    rd = lambda v: pd.read_parquet(ROOT / "submissions" / f"fearless_barn_v{v}.parquet").set_index(config.ID)[config.TARGET].reindex(ids).to_numpy()
    v35, v32, v31 = rd(35), rd(32), rd(31)
    x = sub[config.TARGET].to_numpy()
    L = ap_r == "LIRF"
    print(f"RMS(probe - v35) = {rms(x, v35):.3f} (LIRF {rms(x[L], v35[L]):.2f}, non-LIRF {rms(x[~L], v35[~L]):.2f}) | "
          f"yardsticks v31->v32 {rms(v32, v31):.2f} (-1.79), v32->v34 7.45 (tie)")
    print(f"mean shift vs v35: LIRF {np.mean(x[L]-v35[L]):+.1f} s, non-LIRF {np.mean(x[~L]-v35[~L]):+.1f} s | "
          f"max {x.max()} | >7200 {(x > 7200).sum()} (v35 {(v35 > 7200).sum()}) | validate {errs}")
    scratch = Path(r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
                   r"\9a1d8f93-8788-4b50-8bdc-cd3798959221\scratchpad\season_probe.parquet")
    sub.to_parquet(scratch, index=False)
    if a.version is not None:
        if errs:
            raise SystemExit(f"validation failed: {errs}")
        out = ROOT / "submissions" / f"fearless_barn_v{a.version}.parquet"
        if out.exists():
            raise SystemExit(f"{out} exists -- refusing to overwrite")
        sub.to_parquet(out, index=False)
        (out.parent / (out.name + ".meta.json")).write_text(json.dumps({
            "version": a.version, "built_by": "tests_season_weight/build_probe.py",
            "not_reproducible_via_submit_py": True,
            "note": "board probe of a candidate that FAILED its local bar; see tests_season_weight/PROBE.md",
            "weights_per_airport": W, "mlp": "season-weighted tau=3, 18 seeds, t01/t07 on full (17 epochs)",
            "ftt_seeds": FTT6, "tabnet_seeds": TABNET, "tabnet_w": TW,
            "feature_fingerprint": config.feature_code_fingerprint()}, indent=2))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
