#!/usr/bin/env python3
"""Pre-registered screen for transductive pretraining (PREREGISTERED.md).

The 3-member per-airport weights come FROZEN from
`tests_ftt/solve_vs_strong_mlp.json`: solved on forward_inner with the 12-seed
MLP, never re-solved here. The blend is built twice with those weights, once
with the baseline FTT 3-seed mean and once with the pretrained FTT 3-seed mean
in the same slot, then scored on forward val and test. The seeds are paired
against the baseline standalone.

usage: python tests_ftt_pretrain/screen.py [--tag pt] [--seeds 42,43,44]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import config
from tests_ftt.solve_vs_strong_mlp import surface

rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
CLIP = (config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pt")
    ap.add_argument("--seeds", default="42,43,44")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    frozen = json.loads((ROOT / "tests_ftt" / "solve_vs_strong_mlp.json").read_text())
    W3, mlp_seeds = frozen["w3_per_airport"], frozen["mlp_seeds"]

    base = {s: np.load(ROOT / "tests_ftt" / f"preds_seed{s}_lr1e-3_e132.npz", allow_pickle=True)
            for s in seeds}
    pt = {s: np.load(HERE / f"preds_seed{s}_{a.tag}.npz", allow_pickle=True) for s in seeds}
    for s in seeds:
        for nm, z in (("base", base[s]), ("pt", pt[s])):
            if bool(z["hit_cap"]):
                raise SystemExit(f"{nm} seed {s} hit its cap -- truncated, refusing to screen")
        print(f"seed {s}: base best_ep {int(base[s]['best_epoch'])} val {float(base[s]['val']):.2f} | "
              f"PT best_ep {int(pt[s]['best_epoch'])} val {float(pt[s]['val']):.2f} "
              f"(pretrain best @ {int(pt[s]['pt_best_epoch'])}, capped={bool(pt[s]['pt_hit_cap'])})")

    out = {"tag": a.tag, "seeds": seeds, "w3_source": "tests_ftt/solve_vs_strong_mlp.json (frozen)"}
    for part, key in (("val", "pv"), ("test", "pt")):
        y, apx, tree, mlp = surface("forward", part, mlp_seeds)
        L, ib = apx == "LIRF", y <= 7200
        paired = []
        for s in seeds:
            b, p = base[s][key], pt[s][key]
            paired.append(rm(y[ib], p[ib]) - rm(y[ib], b[ib]))
        blends = {}
        for nm, src in (("ftt", base), ("pt_ftt", pt)):
            f = np.mean([src[s][key] for s in seeds], axis=0)
            three = np.zeros(len(y))
            for k in np.unique(apx):
                m = apx == k
                w = W3.get(str(k), [1 / 3, 1 / 3, 1 / 3])
                three[m] = w[0] * f[m] + w[1] * tree[m] + w[2] * mlp[m]
            three = np.clip(three, *CLIP)
            blends[nm] = {"pooled": rm(y, three), "in_band": rm(y[ib], three[ib]),
                          "lirf": rm(y[L], three[L]), "non_lirf": rm(y[~L], three[~L]),
                          "standalone_in_band": rm(y[ib], f[ib])}
        d = {k: blends["pt_ftt"][k] - blends["ftt"][k] for k in blends["ftt"]}
        print(f"\n===== forward {part.upper()} (n={len(y)}) =====")
        for nm in ("ftt", "pt_ftt"):
            r = blends[nm]
            print(f"  3-member with {nm:7s} pooled {r['pooled']:7.2f} | in-band {r['in_band']:7.2f} | "
                  f"LIRF {r['lirf']:8.2f} | non-LIRF {r['non_lirf']:7.2f} | "
                  f"FTT-ens3 standalone in-band {r['standalone_in_band']:7.2f}")
        print(f"  delta (PT - base): " + " | ".join(f"{k} {v:+.2f}" for k, v in d.items()))
        print(f"  paired standalone in-band deltas per seed: "
              + ", ".join(f"{s}: {v:+.2f}" for s, v in zip(seeds, paired)))
        out[part] = {"blends": blends, "delta": d, "paired_in_band": dict(zip(map(str, seeds), paired))}

    c1 = out["test"]["delta"]["in_band"] <= -1.0
    c2 = out["val"]["delta"]["in_band"] < 0
    c3 = sum(v < 0 for v in out["test"]["paired_in_band"].values()) >= 2
    out["verdict"] = {"test_in_band_le_-1.0": c1, "val_in_band_same_sign": c2,
                      "paired_2_of_3_test": c3, "PASS": bool(c1 and c2 and c3)}
    print(f"\nPRE-REGISTERED BAR: test in-band <= -1.0: {c1} | val same sign: {c2} | "
          f">=2/3 paired seeds improve on test: {c3}  =>  {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
    (HERE / f"screen_{a.tag}.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
