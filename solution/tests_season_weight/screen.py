#!/usr/bin/env python3
"""Pre-registered screen for season-matched per-month MLP weighting."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = range(42, 48)
rm = lambda y, p: float(np.sqrt(np.mean((y - p) ** 2)))


def surface(split, targets):
    m = np.load(HERE / f"meta_{split}.npz", allow_pickle=True)
    y, ap, mon = m["yte"], m["apte"], m["mte"]
    Z = {}
    for mode in ["base"] + [f"t{t:02d}" for t in targets]:
        zs = [np.load(HERE / f"preds_{split}_{mode}_seed{s}.npz") for s in SEEDS]
        capped = [s for s, z in zip(SEEDS, zs) if bool(z["hit_cap"])]
        if capped:
            raise SystemExit(f"{split} {mode}: seeds {capped} hit the cap -- rerun before judging")
        Z[mode] = np.mean([z["pt"] for z in zs], axis=0)
        print(f"  {split} {mode:5s} best epochs {[int(z['best_epoch']) for z in zs]}")
    w = Z["base"].copy()
    for t in targets:
        w[mon == t] = Z[f"t{t:02d}"][mon == t]
    ib, L = y <= 7200, ap == "LIRF"
    out = {"all": (rm(y[ib], Z["base"][ib]), rm(y[ib], w[ib])),
           "lirf": (rm(y[ib & L], Z["base"][ib & L]), rm(y[ib & L], w[ib & L])),
           "pooled": (rm(y, Z["base"]), rm(y, w))}
    for t in targets:
        mm = ib & (mon == t)
        out[f"month{t}"] = (rm(y[mm], Z["base"][mm]), rm(y[mm], w[mm]))
    for k, (b, v) in out.items():
        print(f"  {split} {k:8s} base {b:8.3f} -> weighted {v:8.3f} ({v - b:+.3f})")
    return out


def main() -> int:
    s = surface("seasonal_full", [1, 7])
    f = surface("forward", [11, 12])
    d = lambda r, k: r[k][1] - r[k][0]
    c = {"jan_jul_in_band_le_-1": d(s, "all") <= -1.0,
         "no_month_worse": d(s, "month1") <= 0 and d(s, "month7") <= 0,
         "forward_same_sign": d(f, "all") < 0,
         "lirf_guard": d(s, "lirf") <= 2.0}
    c["PASS"] = all(c.values())
    print("\nPRE-REGISTERED:", c, "=>", "PASS" if c["PASS"] else "FAIL")
    (HERE / "screen.json").write_text(json.dumps({"seasonal_full": s, "forward": f, "verdict": c}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
