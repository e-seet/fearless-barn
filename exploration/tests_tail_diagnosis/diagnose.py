#!/usr/bin/env python3
"""Where does the CURRENT best model's squared error sit on the board's months?

Read-only diagnosis, no training, no selection. Rebuilds the v31 analogue on
seasonal_full TEST (Jan + Jul 2025) from saved predictions:

  3-member per-airport blend  w3 . [FTT-ens3, tree, MLP-ens6]
      (weights solved on forward_inner, frozen; tests_ftt/confirm_seasonal_full.json)
  then TabNet LIRF override   (1 - 0.55) * blend + 0.55 * TabNet-ens3   at LIRF

and splits its squared error by target band, airport, flight-record presence
and month. For each group it reports every component's RMSE and the blend's
mean signed error, so a component that is systematically wrong on a slice of
the tail is visible.

`best_component` columns are DESCRIPTIVE ONLY: choosing a component per group
from this table would be selecting on the surface being reported.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
from common import dataset as D

CACHE = Path(os.environ.get(
    "FTT_CACHE",
    r"C:\Users\Admin\AppData\Local\Temp\claude\C--Users-Admin-Desktop-eddie-fearless-barn"
    r"\2ff7d479-271f-463d-9593-1a05404d65d1\scratchpad\ftt_strong_cache"))
TAB_W = 0.55
BANDS = [(0, 7200, "in-band <=7200"), (7200, 25200, "mid 7.2k-25.2k"),
         (25200, 80000, "high 25.2k-80k"), (80000, np.inf, "artifact >=80k")]


def main() -> int:
    X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
    y = np.asarray(y, float)
    ap = meta["ADEP_mvt"].astype(str).to_numpy()
    cfg = json.loads((ROOT / "tests_ftt" / "confirm_seasonal_full.json").read_text())
    W3 = cfg["w3"]

    tree = np.load(CACHE / "tree_seasonal_full_test.npy")
    mlp = np.load(CACHE / "mlp6_seasonal_full_test.npy")
    ftt = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"]
                   for s in cfg["ftt_seeds"]], axis=0)
    tab = np.mean([np.load(ROOT / "tests_tabnet" / f"preds_seed{s}_sf_lr2e-3p.npz")["pt"]
                   for s in (42, 43, 44)], axis=0)
    assert len(tree) == len(mlp) == len(ftt) == len(tab) == len(y)

    three = np.zeros(len(y))
    for k in np.unique(ap):
        mm = ap == k
        w = W3.get(k, [1 / 3] * 3)
        three[mm] = w[0] * ftt[mm] + w[1] * tree[mm] + w[2] * mlp[mm]
    rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"sanity: 3-member pooled {rm(y, three):.4f} (confirm_seasonal_full.json: {cfg['test']['three']:.4f})")
    assert abs(rm(y, three) - cfg["test"]["three"]) < 0.05, "reconstruction does not match"
    L = ap == "LIRF"
    v31 = three.copy()
    v31[L] = (1 - TAB_W) * three[L] + TAB_W * tab[L]
    print(f"v31 analogue pooled {rm(y, v31):.4f} | in-band {rm(y[y <= 7200], v31[y <= 7200]):.4f}")

    comps = {"v31": v31, "3mem": three, "tree": tree, "mlp6": mlp, "ftt3": ftt, "tab3": tab}
    band = np.full(len(y), "", dtype=object)
    for lo, hi, nm in BANDS:
        band[(y > lo) & (y <= hi)] = nm
    df = pd.DataFrame({"y": y, "ap": ap, "band": band,
                       "rec": np.where(np.asarray(X["has_flight_record"]) > 0, "record", "NO record"),
                       "month": meta["month"].astype(str).to_numpy(),
                       "delta_sched": np.asarray(X["delta_sched_sec"], float)})
    for c, p in comps.items():
        df[c] = p
    total_se = float(((df.y - df.v31) ** 2).sum())

    def table(keys, min_share=0.0):
        rows = []
        for key, g in df.groupby(keys):
            se = float(((g.y - g.v31) ** 2).sum())
            r = {"group": key if isinstance(key, str) else " | ".join(map(str, key)),
                 "n": len(g), "row_share_%": 100 * len(g) / len(df), "SE_share_%": 100 * se / total_se,
                 "v31_rmse": rm(g.y, g.v31), "v31_bias": float((g.v31 - g.y).mean())}
            for c in ("tree", "mlp6", "ftt3", "tab3"):
                r[c] = rm(g.y, g[c])
            best = min(("tree", "mlp6", "ftt3", "tab3", "v31"), key=lambda c: rm(g.y, g[c]))
            r["best_comp(desc)"] = best
            r["SE_if_best_%(desc)"] = 100 * float(((g.y - g[best]) ** 2).sum()) / total_se
            rows.append(r)
        t = pd.DataFrame(rows).sort_values("SE_share_%", ascending=False)
        return t[t["SE_share_%"] >= min_share]

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.2f}".format)
    out = {}
    for name, keys, ms in (("band", ["band"], 0), ("band x record", ["band", "rec"], 0),
                           ("band x month", ["band", "month"], 0),
                           ("band x airport (>= 0.5% of SE)", ["band", "ap"], 0.5)):
        t = table(keys, ms)
        print(f"\n===== {name} =====")
        print(t.to_string(index=False))
        out[name] = t.to_dict(orient="records")

    # LIRF no-record identity check on the tail (the one mechanism on record)
    g = df[(df.ap == "LIRF") & (df.rec == "NO record") & (df.y > 7200)]
    if len(g):
        within = float((np.abs(g.y - g.delta_sched) <= 60).mean())
        print(f"\nLIRF, no record, y > 7200: n={len(g)}, |y - delta_sched| <= 60 s for {100*within:.1f}% "
              f"| v31 rmse {rm(g.y, g.v31):.0f} vs delta_sched rmse {rm(g.y, g.delta_sched):.0f}")
        out["lirf_norec_tail"] = {"n": len(g), "identity_within_60s": within,
                                  "v31_rmse": rm(g.y, g.v31), "delta_sched_rmse": rm(g.y, g.delta_sched)}
    (HERE / "diagnosis.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
