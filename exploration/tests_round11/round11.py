#!/usr/bin/env python3
"""tests_round11/PREREGISTERED.md: Part A (7-airport corrector K vs K2, K3) and Part B (LIRF, LFPG correctors at K)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_round10", ROOT / "tests_round9"):
    sys.path.insert(0, str(p))
import round10 as R10
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
K = R10.PARAMS_K
K2 = {**K, "n_estimators": 3000, "num_leaves": 127, "learning_rate": 0.015}
K3 = {**K, "n_estimators": 3000, "num_leaves": 63, "learning_rate": 0.01}
SEEDS = R10.SEEDS


def corr(trc, X, feat, params):
    return np.mean([lgb.LGBMRegressor(**{**params, "random_state": s}).fit(trc[feat], trc.y - trc.base).predict(X[feat]) for s in SEEDS], axis=0)


def verdict(name, y, P, ref, months, need):
    per = {m: rm(y[m] - P[m]) - rm(y[m] - ref[m]) for m in months}
    changed = [m for m in months if np.any(P[m] != ref[m])]
    wins = sum(per[m] < 0 for m in changed)
    g = np.concatenate([(y[m] - ref[m]) ** 2 - (y[m] - P[m]) ** 2 for m in months]); s0 = np.concatenate([(y[m] - ref[m]) ** 2 for m in months])
    keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
    pooled = (float(np.sqrt(s0.mean())), float(np.sqrt((s0 - g).mean())))
    drop5 = (float(np.sqrt(s0[keep].mean())), float(np.sqrt((s0 - g)[keep].mean())))
    ok = pooled[1] < pooled[0] and wins >= need(len(changed)) and drop5[1] < drop5[0]
    print(f"{name}: pooled {pooled[0]:.3f} -> {pooled[1]:.3f} ({pooled[1] - pooled[0]:+.3f}) | wins {wins}/{len(changed)} | "
          f"best-5 removed {drop5[0]:.3f} -> {drop5[1]:.3f} => {'PASS' if ok else 'FAIL'}", flush=True)
    return {"pooled": pooled, "per_month": per, "wins": wins, "n_months": len(changed), "drop5": drop5, "PASS": bool(ok)}


out = {}
# ---------------- Part A ----------------
df = R10.frame()
months = ["01", "07", "08", "09", "10", "11", "12"]
PA = {k: {} for k in ("K", "K2", "K3")}
for mon in months:
    tr, te = df[(df.mon != mon) & df["cov"]], df[df.mon == mon]
    c = te["cov"].to_numpy(); X = te[c]
    for k, prm in (("K", K), ("K2", K2), ("K3", K3)):
        p = te.base.to_numpy().copy(); p[c] += corr(tr, X, R10.FEAT_A, prm); PA[k][mon] = p
    print(f"  A month {mon}: " + " | ".join(f"{k} {rm(te.y - PA[k][mon]):.3f}" for k in PA), flush=True)
yA = {m: df[df.mon == m].y.to_numpy() for m in months}
out["A"] = {k: verdict(f"A {k} vs K", yA, PA[k], PA["K"], months, lambda n: 5) for k in ("K2", "K3")}

# ---------------- Part B: LIRF ----------------
src = (ROOT / "tests_lirf_adsb" / "lirf_lomo.py").read_text(encoding="utf-8").split("def shuffled")[0]
ns = {"__file__": str(ROOT / "tests_lirf_adsb" / "lirf_lomo.py"), "__name__": "lirf"}
exec(compile(src, "lirf_frame", "exec"), ns)
L = ns["df"]; FEAT_L = ns["FEAT"]; PARAMS_OLD = ns["PARAMS"]
lm = sorted(L.mon.unique()); PL = {"old": {}, "K": {}}
for mon in lm:
    tr, te = L[(L.mon != mon) & L.elig], L[L.mon == mon]
    c = te.elig.to_numpy()
    for k, prm in (("old", PARAMS_OLD), ("K", K)):
        p = te.base.to_numpy().copy(); p[c] += corr(tr, te[c], FEAT_L, prm); PL[k][mon] = p
    print(f"  LIRF month {mon}: old {rm(te.y - PL['old'][mon]):.2f} | K {rm(te.y - PL['K'][mon]):.2f}", flush=True)
out["LIRF"] = verdict("B LIRF K vs v55 setting", {m: L[L.mon == m].y.to_numpy() for m in lm}, PL["K"], PL["old"], lm, lambda n: 4)

# ---------------- Part B: LFPG ----------------
import round9 as R9
D9 = R9.frame()
PF = {"old": {}, "K": {}}
for mon in months:
    tr, te = D9[D9.mon != mon], D9[D9.mon == mon]
    trc = tr[R9.lfpg(tr)]; c = R9.lfpg(te).to_numpy()
    for k, prm in (("old", R9.PARAMS), ("K", K)):
        p = te.base.to_numpy().copy()
        if c.any() and len(trc) >= 200:
            p[c] += corr(trc, te[c], R9.FEAT_1, prm)
        PF[k][mon] = p
    print(f"  LFPG month {mon}: old {rm(te.y - PF['old'][mon]):.3f} | K {rm(te.y - PF['K'][mon]):.3f}", flush=True)
out["LFPG"] = verdict("B LFPG K vs v56 setting", {m: D9[D9.mon == m].y.to_numpy() for m in months}, PF["K"], PF["old"], months, lambda n: 2 / 3 * n)
(HERE / "round11.json").write_text(json.dumps(out, indent=2, default=float))
print("ALL DONE")
