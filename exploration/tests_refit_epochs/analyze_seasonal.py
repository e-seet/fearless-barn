#!/usr/bin/env python3
"""Pre-registered confirmation on seasonal_full (PREREGISTERED_seasonal.md).

usage: python tests_refit_epochs/analyze_seasonal.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
from tests_refit_epochs.analyze import chosen_epoch, rm

SEEDS = list(range(42, 54))
CACHE = Path(os.environ.get(
    "FTT_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
    r"\2ff7d479-271f-463d-9593-1a05404d65d1\scratchpad\ftt_strong_cache"))


def main() -> int:
    m = np.load(HERE / "meta_seasonal.npz", allow_pickle=True)
    cur = {s: np.load(HERE / f"curves_seasonal_seed{s}.npz") for s in SEEDS}
    yd, yt, apt = m["y_dec"], m["y_janjul"], m["ap_janjul"]
    ibd, ibt, L = yd <= 7200, yt <= 7200, apt == "LIRF"
    n_ep = cur[SEEDS[0]]["S_dec"].shape[0]

    own = {s: chosen_epoch([rm(yd, p) for p in cur[s]["S_dec"]]) for s in SEEDS}
    ship = {s: own[42] if s <= 51 else own[44] for s in SEEDS}
    dec_curve = np.array([rm(yd[ibd], np.mean([cur[s]["S_dec"][e][ibd] for s in SEEDS], axis=0))
                          for e in range(n_ep)])
    roll = np.array([dec_curve[max(0, e - 2):e + 3].mean() for e in range(n_ep)])
    E_ens = int(np.argmin(roll)) + 1
    ens = {s: E_ens for s in SEEDS}
    print(f"own epochs {own} (median {np.median(list(own.values()))})")
    print(f"ens rule: 5-epoch centred rolling argmin on Dec ensemble in-band -> E = {E_ens}")
    print("Dec ensemble in-band curve: " + " ".join(f"{e+1}:{dec_curve[e]:.1f}" for e in range(n_ep)))

    def score(eps):
        p = np.mean([cur[s]["S_janjul"][eps[s] - 1] for s in SEEDS], axis=0)
        return {"in_band": rm(yt[ibt], p[ibt]), "lirf_in_band": rm(yt[ibt & L], p[ibt & L]),
                "nonlirf_in_band": rm(yt[ibt & ~L], p[ibt & ~L]), "pooled": rm(yt, p),
                "n_over_7200": int((p > 7200).sum()), "max_pred": float(p.max()),
                "epochs": sorted(set(eps.values()))}

    res = {k: score(v) for k, v in (("own", own), ("ship", ship), ("ens", ens))}
    print("\nJan+Jul, 12-seed ensemble:")
    for k, r in res.items():
        print(f"  {k:5s} epochs {str(r['epochs']):28s} in-band {r['in_band']:8.3f} | LIRF {r['lirf_in_band']:8.3f} "
              f"| non-LIRF {r['nonlirf_in_band']:8.3f} | pooled {r['pooled']:8.3f} | >7200 {r['n_over_7200']} "
              f"| max {r['max_pred']:.0f}")
    e, o, sh = res["ens"], res["own"], res["ship"]
    c1 = e["in_band"] <= o["in_band"] - 2.0 and e["in_band"] <= sh["in_band"] - 2.0
    c2 = e["lirf_in_band"] - o["lirf_in_band"] <= 2.0
    c3 = e["pooled"] - o["pooled"] <= 2.0 and e["n_over_7200"] <= 1.5 * max(o["n_over_7200"], 1)
    verdict = "PASS" if (c1 and c2 and c3) else "FAIL"
    print(f"\nPRE-REGISTERED BAR: in-band >= 2.0 better than own AND ship: {c1} "
          f"(vs own {e['in_band']-o['in_band']:+.3f}, vs ship {e['in_band']-sh['in_band']:+.3f}) | "
          f"LIRF guard: {c2} ({e['lirf_in_band']-o['lirf_in_band']:+.3f}) | "
          f"tail guard: {c3} (pooled {e['pooled']-o['pooled']:+.3f}, >7200 {e['n_over_7200']} vs {o['n_over_7200']})"
          f"  =>  {verdict}")

    # ---- descriptive: inside the tree+MLP blend, alphas from forward_inner at matching E ----
    blend = {}
    tree_t = CACHE / "tree_seasonal_full_test.npy"
    mA = HERE / "meta.npz"
    if tree_t.exists() and mA.exists():
        from pipeline_airport_operator_multimodel.train import solve_alpha_per_airport
        tt, ma = np.load(tree_t), np.load(mA, allow_pickle=True)
        tree_ja = np.load(CACHE / "tree_forward_inner_val.npy")
        A = {s: np.load(HERE / f"curves_seed{s}.npz")["A_julaug"] for s in SEEDS}
        if len(tt) == len(yt):
            print("\nDESCRIPTIVE blend (alpha solved on Jul-Aug at the same uniform E, applied to Jan+Jul):")
            for k, eps in (("own_median", {s: int(np.ceil(np.median(list(own.values())))) for s in SEEDS}),
                           ("ens", ens)):
                E = eps[SEEDS[0]]
                al = solve_alpha_per_airport(ma["y_julaug"], tree_ja,
                                             np.mean([A[s][min(E, A[s].shape[0]) - 1] for s in SEEDS], axis=0),
                                             ma["ap_julaug"])
                mlp = np.mean([cur[s]["S_janjul"][E - 1] for s in SEEDS], axis=0)
                a = np.array([al.get(str(x), 0.5) for x in apt])
                p = a * tt + (1 - a) * mlp
                blend[k] = {"E": E, "in_band": rm(yt[ibt], p[ibt]), "lirf": rm(yt[ibt & L], p[ibt & L]),
                            "nonlirf": rm(yt[ibt & ~L], p[ibt & ~L]), "pooled": rm(yt, p)}
                print(f"  {k:10s} E={E:2d} in-band {blend[k]['in_band']:.3f} | LIRF {blend[k]['lirf']:.3f} | "
                      f"non-LIRF {blend[k]['nonlirf']:.3f} | pooled {blend[k]['pooled']:.3f}")
        else:
            print(f"\n(blend skipped: cached tree has {len(tt)} rows, Jan+Jul has {len(yt)})")

    (HERE / "analysis_seasonal.json").write_text(json.dumps(
        {"own": own, "E_ens": E_ens, "dec_curve": dec_curve.tolist(), "results": res,
         "bar": {"in_band": c1, "lirf_guard": c2, "tail_guard": c3, "verdict": verdict},
         "blend_descriptive": blend}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
