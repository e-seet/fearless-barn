#!/usr/bin/env python3
"""FTT downward-only guard at LIRF (PREREGISTERED.md). No training.

    P_new = P - w * max(0, P - FTT),  P = 0.45*MLP + 0.55*TabNet  (LIRF rows)
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
C = Path(os.environ.get("FTT_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
    r"\2ff7d479-271f-463d-9593-1a05404d65d1\scratchpad\ftt_strong_cache"))
TW, FTT_SEEDS, TAB = 0.55, range(42, 48), (42, 43, 44)
F = ROOT / "tests_ftt"
T = ROOT / "tests_tabnet"
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))
ftt = lambda path_fn, key: np.mean([np.load(path_fn(s), allow_pickle=True)[key] for s in FTT_SEEDS], axis=0)


def surfaces():
    mi = np.load(F / "meta_forward_inner.npz", allow_pickle=True)
    mf = np.load(F / "meta.npz", allow_pickle=True)
    ms = np.load(F / "meta_seasonal_full.npz", allow_pickle=True)
    fw = lambda s: F / (f"preds_seed{s}_lr1e-3_e132.npz" if s <= 44 else f"preds_seed{s}_lr1e-3_e175.npz")
    yield ("forward_inner", mi["yva"], mi["apva"], np.load(C / "mlp18_forward_inner_val.npy"),
           np.mean([np.load(T / "build" / f"inner_seed{s}.npz")["pv"] for s in TAB], axis=0),
           ftt(lambda s: F / f"preds_seed{s}_forward_inner_lr1e-3.npz", "pv"))
    for part, key, y, ap in (("forward_val", "pv", mf["yva"], mf["apva"]), ("forward_test", "pt", mf["yte"], mf["apte"])):
        yield (part, y, ap, np.load(C / f"mlp12_{part}.npy"),
               np.mean([np.load(T / f"preds_seed{s}_lr2e-3p.npz")[key] for s in TAB], axis=0), ftt(fw, key))
    for part, key, y, ap in (("seasonal_val", "pv", ms["yva"], ms["apva"]), ("seasonal_test", "pt", ms["yte"], ms["apte"])):
        yield (part, y, ap, np.load(C / f"mlp6_seasonal_full_{'val' if key == 'pv' else 'test'}.npy"),
               np.mean([np.load(T / f"preds_seed{s}_sf_lr2e-3p.npz")[key] for s in TAB], axis=0),
               ftt(lambda s: F / f"preds_seed{s}_seasonal_full_lr1e-3.npz", key))


def main() -> int:
    S = {}
    for name, y, ap, mlp, tab, f in surfaces():
        assert len(y) == len(mlp) == len(tab) == len(f), name
        L = ap == "LIRF"
        P = (1 - TW) * mlp[L] + TW * tab[L]
        S[name] = (y[L], P, np.maximum(0.0, P - f[L]))
    y, P, D = S["forward_inner"]
    w = float(np.clip(np.dot(P - y, D) / max(np.dot(D, D), 1e-9), 0.0, 1.0))   # min ||y - (P - wD)||^2
    frac = float((D > 0).mean())
    print(f"w solved on forward_inner LIRF = {w:.4f}  (FTT below P on {100*frac:.1f}% of LIRF rows)")
    out = {"w": w, "surfaces": {}}
    for name, (y, P, D) in S.items():
        Pn = P - w * D
        ib = y <= 7200
        r = {"in_band": (rm(y[ib], P[ib]), rm(y[ib], Pn[ib])), "pooled": (rm(y, P), rm(y, Pn)), "n": int(len(y))}
        out["surfaces"][name] = r
        tag = " (solve surface, in-sample)" if name == "forward_inner" else ""
        print(f"  {name:14s} LIRF in-band {r['in_band'][0]:8.2f} -> {r['in_band'][1]:8.2f} ({r['in_band'][1]-r['in_band'][0]:+.3f}) | "
              f"pooled {r['pooled'][0]:8.2f} -> {r['pooled'][1]:8.2f} ({r['pooled'][1]-r['pooled'][0]:+.3f}){tag}")
    s = out["surfaces"]
    d_ib = {k: s[k]["in_band"][1] - s[k]["in_band"][0] for k in ("forward_val", "forward_test", "seasonal_test")}
    d_po = {k: s[k]["pooled"][1] - s[k]["pooled"][0] for k in ("forward_test", "seasonal_test")}
    c1, c2, c3 = w > 0.05, all(v < 0 for v in d_ib.values()), all(v <= 2.0 for v in d_po.values())
    out["verdict"] = {"w_gt_0.05": c1, "in_band_better_all_three": c2, "pooled_tail_guard": c3,
                      "PASS": bool(c1 and c2 and c3)}
    print(f"\nPRE-REGISTERED: w > 0.05 {c1} | in-band better on fwd val/test + Jan+Jul {c2} {d_ib} | "
          f"pooled guard {c3} {d_po}  =>  {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
    (HERE / "result.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
