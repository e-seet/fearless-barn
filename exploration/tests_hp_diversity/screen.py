#!/usr/bin/env python3
"""Pre-registered screen for hyperparameter-diverse MLP members (PREREGISTERED.md).

usage: python tests_hp_diversity/screen.py [--variants wide,deep,reg,bigbatch]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="wide,deep,reg,bigbatch")
    ap.add_argument("--split", default="forward", choices=["forward", "seasonal_full"])
    a = ap.parse_args(argv)
    variants = [v for v in a.variants.split(",") if v]
    sfx = "" if a.split == "forward" else "_" + a.split
    pre = "" if a.split == "forward" else a.split + "_"
    m = np.load(HERE / f"meta{sfx}.npz", allow_pickle=True)
    Y = {"val": m["yva"], "test": m["yte"]}
    IB = {k: Y[k] <= 7200 for k in Y}
    key = {"val": "pv", "test": "pt"}

    def load(c, s):
        z = np.load(HERE / f"preds_{pre}{c}_seed{s}.npz", allow_pickle=True)
        if bool(z["hit_cap"]):
            raise SystemExit(f"{c} seed {s} hit its cap -- refusing to screen")
        return z

    S1, S2 = [42, 43, 44], [45, 46, 47]
    base = {s: load("base", s) for s in S1 + S2}
    print("config   seed: best_ep / val")
    for c in ["base"] + variants:
        zs = base if c == "base" else {s: load(c, s) for s in S1 + S2}
        print(f"  {c:9s} " + "  ".join(f"{s}:{int(z['best_epoch'])}/{float(z['val']):.1f}"
                                        for s, z in zs.items()))

    ens = lambda zs, part: np.mean([z[key[part]] for z in zs], axis=0)
    out = {}
    for part in ("val", "test"):
        y, ib = Y[part], IB[part]
        A = ens([base[s] for s in S1 + S2], part)
        b1, b2 = ens([base[s] for s in S1], part), ens([base[s] for s in S2], part)
        ref_corr = float(np.corrcoef(y[ib] - b1[ib], y[ib] - b2[ib])[0, 1])
        print(f"\n===== {a.split} {part.upper()} | A (6 base) in-band {rm(y[ib], A[ib]):.3f} | "
              f"base-ens3 vs base-ens3 residual corr {ref_corr:.4f} (the no-diversity reference) =====")
        out[part] = {"A_in_band": rm(y[ib], A[ib]), "ref_corr": ref_corr, "variants": {}}
        for v in variants:
            vz = {s: load(v, s) for s in S1 + S2}
            B1 = ens([base[s] for s in S1] + [vz[s] for s in S2], part)
            B2 = ens([base[s] for s in S2] + [vz[s] for s in S1], part)
            v3a, v3b = ens([vz[s] for s in S1], part), ens([vz[s] for s in S2], part)
            corr = float(np.mean([np.corrcoef(y[ib] - vv[ib], y[ib] - bb[ib])[0, 1]
                                  for vv, bb in ((v3a, b2), (v3b, b1))]))
            d1, d2 = rm(y[ib], B1[ib]) - rm(y[ib], A[ib]), rm(y[ib], B2[ib]) - rm(y[ib], A[ib])
            stand = float(np.mean([rm(y[ib], v3a[ib]), rm(y[ib], v3b[ib])]))
            stand_b = float(np.mean([rm(y[ib], b1[ib]), rm(y[ib], b2[ib])]))
            print(f"  {v:9s} B1-A {d1:+.3f} | B2-A {d2:+.3f} | mean {np.mean([d1, d2]):+.3f} | "
                  f"resid corr vs base {corr:.4f} | ens3 standalone {stand:.2f} (base ens3 {stand_b:.2f})")
            out[part]["variants"][v] = {"d_B1": d1, "d_B2": d2, "d_mean": float(np.mean([d1, d2])),
                                        "corr": corr, "standalone_ens3": stand, "base_ens3": stand_b}

    print("\nPRE-REGISTERED BAR (per variant): test mean <= -1.0 AND both B1,B2 < A on test AND val mean < 0")
    verdict = {}
    for v in variants:
        t, vv = out["test"]["variants"][v], out["val"]["variants"][v]
        c1, c2, c3 = t["d_mean"] <= -1.0, (t["d_B1"] < 0 and t["d_B2"] < 0), vv["d_mean"] < 0
        verdict[v] = {"test_mean_le_-1": c1, "both_halves_test": c2, "val_same_sign": c3,
                      "PASS": bool(c1 and c2 and c3)}
        print(f"  {v:9s} {c1} {c2} {c3}  =>  {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
    out["verdict"] = verdict
    (HERE / f"screen{sfx}.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
