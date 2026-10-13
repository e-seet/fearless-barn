#!/usr/bin/env python3
"""Pre-registered screen for the backbone variants (PREREGISTERED.md).

usage: python tests_backbone/screen.py [--variants ple,tabm,tabm_ple]
Stage 1 (forward) always runs; stage 2 (seasonal_full) runs when its files exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46, 47]
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def load(pre, c, s):
    z = np.load(HERE / f"preds_{pre}{c}_seed{s}.npz", allow_pickle=True)
    if bool(z["hit_cap"]):
        raise SystemExit(f"{pre}{c} seed {s} hit its cap -- unresolved, refusing to judge")
    return z


def stage(split, variants):
    pre, sfx = ("", "") if split == "forward" else (split + "_", "_" + split)
    if not all((HERE / f"preds_{pre}{c}_seed{s}.npz").exists() for c in ["base"] + variants for s in SEEDS):
        print(f"\n[{split}] not all runs present yet")
        return None
    m = np.load(HERE / f"meta{sfx}.npz", allow_pickle=True)
    Z = {c: {s: load(pre, c, s) for s in SEEDS} for c in ["base"] + variants}
    print(f"\n===== {split} =====  (best_ep/val per seed)")
    for c, zs in Z.items():
        print(f"  {c:9s} " + "  ".join(f"{s}:{int(z['best_epoch'])}/{float(z['val']):.1f}" for s, z in zs.items()))
    res = {}
    for part, key, y, ap in (("val", "pv", m["yva"], m["apva"]), ("test", "pt", m["yte"], m["apte"])):
        ib, L = y <= 7200, ap == "LIRF"
        ens = {c: np.mean([Z[c][s][key] for s in SEEDS], axis=0) for c in Z}
        r = {c: {"in_band": rm(y[ib], e[ib]), "lirf_in_band": rm(y[ib & L], e[ib & L]),
                 "pooled": rm(y, e)} for c, e in ens.items()}
        for v in variants:
            r[v]["d_in_band"] = r[v]["in_band"] - r["base"]["in_band"]
            r[v]["d_lirf"] = r[v]["lirf_in_band"] - r["base"]["lirf_in_band"]
            r[v]["paired_better"] = int(sum(rm(y[ib], Z[v][s][key][ib]) < rm(y[ib], Z["base"][s][key][ib])
                                            for s in SEEDS))
        print(f"  {part.upper()}: base ens6 in-band {r['base']['in_band']:.3f} | LIRF {r['base']['lirf_in_band']:.2f} | pooled {r['base']['pooled']:.2f}")
        for v in variants:
            q = r[v]
            print(f"    {v:9s} in-band {q['in_band']:.3f} ({q['d_in_band']:+.3f}) | LIRF {q['lirf_in_band']:.2f} "
                  f"({q['d_lirf']:+.2f}) | pooled {q['pooled']:.2f} | paired better {q['paired_better']}/6")
        res[part] = r
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="ple,tabm,tabm_ple")
    a = ap.parse_args(argv)
    variants = [v for v in a.variants.split(",") if v]
    s1, s2 = stage("forward", variants), stage("seasonal_full", variants)
    verdict = {}
    for v in variants:
        p1 = s1 is not None and (s1["test"][v]["d_in_band"] <= -1.0 and s1["val"][v]["d_in_band"] < 0
                                 and s1["test"][v]["paired_better"] >= 4)
        p2 = None if s2 is None else (s2["test"][v]["d_in_band"] <= -1.0 and s2["val"][v]["d_in_band"] < 0
                                      and s2["test"][v]["d_lirf"] <= 2.0)
        verdict[v] = {"stage1_forward": p1, "stage2_seasonal": p2,
                      "CANDIDATE": bool(p1 and p2)}
    print("\nPRE-REGISTERED VERDICT:")
    for v, d in verdict.items():
        print(f"  {v:9s} stage1 {d['stage1_forward']} | stage2 {d['stage2_seasonal']} | "
              f"{'CANDIDATE' if d['CANDIDATE'] else ('PENDING' if d['stage2_seasonal'] is None and d['stage1_forward'] else 'not a candidate')}")
    (HERE / "screen.json").write_text(json.dumps({"forward": s1, "seasonal_full": s2, "verdict": verdict},
                                                 indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
