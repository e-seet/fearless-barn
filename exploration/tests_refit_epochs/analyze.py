#!/usr/bin/env python3
"""Apply the pre-registered rules (PREREGISTERED.md) to the saved curves.

usage: python tests_refit_epochs/analyze.py [--seeds 42,...,53]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

PATIENCE, CAP = 6, 45
REGISTERED_FORWARD_BEST = {42: 10, 43: 19, 44: 14, 45: 10, 46: 11, 47: 10, 48: 11,
                           49: 11, 50: 15, 51: 16, 52: 21, 53: 20}
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def chosen_epoch(rmse_by_epoch):
    """Production early stopping: patience 6, cap 45, 1-indexed best epoch."""
    best, best_ep, bad = 1e18, 0, 0
    for i, r in enumerate(rmse_by_epoch[:CAP], start=1):
        if r < best:
            best, best_ep, bad = r, i, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_ep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=",".join(str(s) for s in range(42, 54)))
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    meta = np.load(HERE / "meta.npz", allow_pickle=True)
    cur = {s: np.load(HERE / f"curves_seed{s}.npz") for s in seeds}
    y = {k: meta[f"y_{k}"] for k in ("julaug", "sepoct", "novdec")}
    ib = {k: y[k] <= 7200 for k in y}
    nA, nB, nC = (int(meta[f"n_train_{s}"]) for s in "ABC")

    E = {"A": {}, "B": {}}
    for s in seeds:
        E["A"][s] = chosen_epoch([rm(y["julaug"], p) for p in cur[s]["A_julaug"]])
        E["B"][s] = chosen_epoch([rm(y["sepoct"], p) for p in cur[s]["B_sepoct"]])
    print("chosen epochs (patience 6, cap 45):")
    print(f"  A (Jan-Jun -> Jul-Aug): {E['A']}  median {np.median(list(E['A'].values()))}")
    print(f"  B (Jan-Aug -> Sep-Oct): {E['B']}  median {np.median(list(E['B'].values()))}")
    print(f"  registered forward bests (CPU pipeline): {[REGISTERED_FORWARD_BEST[s] for s in seeds]}"
          f"  median {np.median([REGISTERED_FORWARD_BEST[s] for s in seeds])}")

    surfaces = {
        "surface1": dict(src="B", tgt_key="C_novdec", ev="novdec", ratio=nC / nB,
                         desc="E from B (Sep-Oct) -> applied to C, scored Nov-Dec"),
        "surface2": dict(src="A", tgt_key="B_sepoct", ev="sepoct", ratio=nB / nA,
                         desc="E from A (Jul-Aug) -> applied to B, scored Sep-Oct"),
    }
    out = {"seeds": seeds, "chosen": {k: {str(s): v for s, v in d.items()} for k, d in E.items()},
           "n_train": {"A": nA, "B": nB, "C": nC}}
    for name, sf in surfaces.items():
        Es = E[sf["src"]]
        med = float(np.median(list(Es.values())))
        n_ep = cur[seeds[0]][sf["tgt_key"]].shape[0]
        clampE = lambda e: int(min(max(e, 1), n_ep))
        rules = {
            "ship": {s: clampE(Es[42] if s <= 51 else Es[44]) for s in seeds},
            "median": {s: clampE(math.ceil(med)) for s in seeds},
            "div": {s: clampE(round(med / sf["ratio"])) for s in seeds},
            "mul": {s: clampE(round(med * sf["ratio"])) for s in seeds},
            "own": {s: clampE(Es[s]) for s in seeds},
        }
        yy, m = y[sf["ev"]], ib[sf["ev"]]
        # uniform-E ensemble curve (for the oracle, descriptive)
        ens_curve = [rm(yy[m], np.mean([cur[s][sf["tgt_key"]][e][m] for s in seeds], axis=0))
                     for e in range(n_ep)]
        oracle = int(np.argmin(ens_curve)) + 1
        res = {}
        for r, eps in rules.items():
            ens = np.mean([cur[s][sf["tgt_key"]][eps[s] - 1] for s in seeds], axis=0)
            single = {s: rm(yy[m], cur[s][sf["tgt_key"]][eps[s] - 1][m]) for s in seeds}
            res[r] = {"epochs": sorted(set(eps.values())), "ens_in_band": rm(yy[m], ens[m]),
                      "ens_pooled": rm(yy, ens), "single_in_band": single}
        print(f"\n===== {name}: {sf['desc']} | row ratio {sf['ratio']:.3f} | median E {med} =====")
        for r, v in res.items():
            d = v["ens_in_band"] - res["ship"]["ens_in_band"]
            paired = [v["single_in_band"][s] - res["ship"]["single_in_band"][s] for s in seeds]
            print(f"  {r:7s} epochs {str(v['epochs']):12s} ens in-band {v['ens_in_band']:8.3f} "
                  f"(vs ship {d:+.3f}) | ens pooled {v['ens_pooled']:8.3f} | single paired vs ship: "
                  f"mean {np.mean(paired):+.2f}, better in {sum(p < 0 for p in paired)}/{len(seeds)}")
        print(f"  oracle (DESCRIPTIVE ONLY): uniform E={oracle}, ens in-band {min(ens_curve):.3f}")
        print("  uniform-E ensemble in-band curve: " +
              " ".join(f"{e+1}:{ens_curve[e]:.1f}" for e in range(0, min(n_ep, 40))))
        out[name] = {"ratio": sf["ratio"], "median_E": med, "rules": res,
                     "oracle_E": oracle, "ens_curve_in_band": ens_curve}

    # ---- pre-registered decision ----
    s1, s2 = out["surface1"]["rules"], out["surface2"]["rules"]
    ib1 = {r: s1[r]["ens_in_band"] for r in s1}
    ib2 = {r: s2[r]["ens_in_band"] for r in s2}
    median_ok = ib1["median"] - ib1["ship"] <= 0.5
    directional = {}
    for r in ("div", "mul"):
        directional[r] = all(ibx[r] <= min(ibx["ship"], ibx["median"]) - 1.0 for ibx in (ib1, ib2))
    if any(directional.values()):
        verdict = "ADOPT " + [r for r, ok in directional.items() if ok][0]
    elif median_ok:
        verdict = "ADOPT median (default)"
    else:
        verdict = "KEEP production 10/14 (median worse than ship by > 0.5 s on surface 1)"
    out["decision"] = {"median_within_0.5_of_ship_s1": median_ok, "directional_pass": directional,
                       "verdict": verdict}
    print(f"\nPRE-REGISTERED DECISION: median within 0.5 of ship on s1: {median_ok} | "
          f"directional passes: {directional}  =>  {verdict}")

    # ---- descriptive: does adding Sep-Oct to training help at all? ----
    yn, mn = y["novdec"], ib["novdec"]
    b_own = np.mean([cur[s]["B_novdec"][E["B"][s] - 1] for s in seeds], axis=0)
    c_own = np.mean([cur[s]["C_novdec"][E["B"][s] - 1] for s in seeds], axis=0)
    print(f"\nDESCRIPTIVE refit sanity (Nov-Dec, each seed at its B-chosen epoch): "
          f"B (Jan-Aug) ens in-band {rm(yn[mn], b_own[mn]):.3f} vs C (Jan-Oct) {rm(yn[mn], c_own[mn]):.3f}")
    out["refit_sanity"] = {"B_ens_in_band": rm(yn[mn], b_own[mn]), "C_ens_in_band": rm(yn[mn], c_own[mn])}

    (HERE / "analysis.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
