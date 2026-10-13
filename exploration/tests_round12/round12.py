#!/usr/bin/env python3
"""tests_round12/PREREGISTERED.md: corrector trained on the equal-weight base (A) vs on the realistic per-airport-weighted
base (B), both applied to the realistic base. Leave-one-month-out over 7 months, K2 setting, seeds 42-46."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_round10"):
    sys.path.insert(0, str(p))
import round10 as R10
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
K2 = {**R10.PARAMS_K, "n_estimators": 3000, "num_leaves": 127, "learning_rate": 0.015}
W = json.loads((ROOT / "tests_ftt" / "final_weights_c3d6.json").read_text())["weights_per_airport"]

df = R10.frame()
tree, mlp, ftt = df.base + df.d_tree, df.base + df.d_mlp, df.base + df.d_ftt
w = np.array([W.get(a, [1 / 3] * 3) for a in df.ap])
df["base_eq"] = df.base
df["base_real"] = w[:, 0] * ftt + w[:, 1] * tree + w[:, 2] * mlp
print(f"{len(df)} rows | equal-base RMSE {rm(df.y - df.base_eq):.2f} | realistic-base RMSE {rm(df.y - df.base_real):.2f} | "
      f"mean |real - equal| on covered rows {np.abs(df.base_real - df.base_eq)[df['cov']].mean():.1f} s", flush=True)


def corr(trc, bcol, X):
    t = trc.copy(); t["base"] = t[bcol]
    return np.mean([lgb.LGBMRegressor(**{**K2, "random_state": s}).fit(t[R10.FEAT_A], t.y - t.base).predict(X[R10.FEAT_A]) for s in R10.SEEDS], axis=0)


months = ["01", "07", "08", "09", "10", "11", "12"]
P = {"base": {}, "A": {}, "B": {}}
for mon in months:
    tr, te = df[(df.mon != mon) & df["cov"]], df[df.mon == mon].copy()
    te["base"] = te.base_real
    c = te["cov"].to_numpy(); X = te[c]
    P["base"][mon] = te.base_real.to_numpy()
    for k, bcol in (("A", "base_eq"), ("B", "base_real")):
        p = te.base_real.to_numpy().copy(); p[c] += corr(tr, bcol, X); P[k][mon] = p
    print(f"  month {mon}: realistic base {rm(te.y - P['base'][mon]):.3f} | A {rm(te.y - P['A'][mon]):.3f} | B {rm(te.y - P['B'][mon]):.3f} "
          f"({rm(te.y - P['B'][mon]) - rm(te.y - P['A'][mon]):+.3f})", flush=True)
y = {m: df[df.mon == m].y.to_numpy() for m in months}
pooled = {k: float(np.sqrt(np.concatenate([(y[m] - P[k][m]) ** 2 for m in months]).mean())) for k in P}
wins = sum(rm(y[m] - P["B"][m]) < rm(y[m] - P["A"][m]) for m in months)
g = np.concatenate([(y[m] - P["A"][m]) ** 2 - (y[m] - P["B"][m]) ** 2 for m in months]); sA = np.concatenate([(y[m] - P["A"][m]) ** 2 for m in months])
keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
drop5 = (float(np.sqrt(sA[keep].mean())), float(np.sqrt((sA - g)[keep].mean())))
ok = pooled["B"] < pooled["A"] and wins >= 5 and drop5[1] < drop5[0]
print(f"\nPOOLED: realistic base {pooled['base']:.3f} | A {pooled['A']:.3f} | B {pooled['B']:.3f} ({pooled['B'] - pooled['A']:+.3f}) | "
      f"{wins}/7 months | best-5 removed {drop5[0]:.3f} -> {drop5[1]:.3f} => {'PASS' if ok else 'FAIL'}")
(HERE / "round12.json").write_text(json.dumps({"pooled": pooled, "wins": wins, "drop5": drop5, "PASS": bool(ok)}, indent=2))
print("ALL DONE")
