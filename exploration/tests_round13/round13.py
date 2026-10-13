#!/usr/bin/env python3
"""tests_round13/PREREGISTERED.md: stage-2 B (v42-weight base) vs C (v59 4-member base); LFPG L0 (equal base) vs L1 (4-member base)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_round10", ROOT / "tests_round9"):
    sys.path.insert(0, str(p))
import round10 as R10
import round9 as R9
from common import dataset as D
from common import io as cio
from common.predict import predict_arm
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
K2 = {**R10.PARAMS_K, "n_estimators": 3000, "num_leaves": 127, "learning_rate": 0.015}
W3 = json.loads((ROOT / "tests_ftt" / "final_weights_c3d6.json").read_text())["weights_per_airport"]
W4 = json.loads((ROOT / "tests_ftt_wide" / "final_weights_v59_s4.json").read_text())["weights_per_airport"]
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)

# held-out CatBoost per surface, joined by MVT_ID
cat_parts = []
for split, part, seeds in (("seasonal_full", "test", (42,)), ("forward_inner", "val", (42, 43, 44, 45, 46, 47)),
                           ("forward", "val", (42, 43, 44, 45, 46, 47)), ("forward", "test", (42, 43, 44, 45, 46, 47))):
    f = CACHE / f"cb1_{split}_{part}.parquet"
    if not f.exists():
        Xtr, _, _ = D.load_part(split, "train", "with_aobt", filter_target=True, outlier_policy="keep_all")
        X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
        p = np.mean([predict_arm("catboost_antigravity", split, part, "with_aobt", s, "keep_all", X=X, meta=meta, Xtr=Xtr, tag="cb1") for s in seeds], axis=0)
        pd.DataFrame({"cat": p}, index=X.index).to_parquet(f)
    cat_parts.append(pd.read_parquet(f))
cat = pd.concat(cat_parts); cat = cat[~cat.index.duplicated()]
print("held-out CatBoost predictions:", len(cat), flush=True)

df = R10.frame()
df = df.join(cat)
assert df["cat"].notna().all(), "missing CatBoost held-out predictions"
rec = cio.load_training(columns=["MVT_ID_mvt", "FLIGHT_ID_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").FLIGHT_ID_mvt.notna()
df["rec"] = rec.reindex(df.index).fillna(False).to_numpy()
tree, mlp, ftt = df.base + df.d_tree, df.base + df.d_mlp, df.base + df.d_ftt
w3 = np.array([W3.get(a, [1 / 3] * 3) for a in df.ap]); w4 = np.array([W4.get(a, [0.25] * 4) for a in df.ap])
df["base_eq"] = df.base
df["base_r3"] = w3[:, 0] * ftt + w3[:, 1] * tree + w3[:, 2] * mlp
df["base_r4"] = w4[:, 0] * ftt + w4[:, 1] * tree + w4[:, 2] * mlp + w4[:, 3] * df["cat"]
print(f"base RMSE: equal {rm(df.y - df.base_eq):.2f} | r3 {rm(df.y - df.base_r3):.2f} | r4 {rm(df.y - df.base_r4):.2f}", flush=True)
df["lfpg"] = df.ap.eq("LFPG") & df.e_first.notna() & df.rec


def corr(trc, bcol, X, feat, params):
    t = trc.copy(); t["base"] = t[bcol]
    return np.mean([lgb.LGBMRegressor(**{**params, "random_state": s}).fit(t[feat], t.y - t.base).predict(X[feat]) for s in R10.SEEDS], axis=0)


months = ["01", "07", "08", "09", "10", "11", "12"]
P = {k: {} for k in ("base", "B", "C", "L0", "L1")}
for mon in months:
    tr, te = df[df.mon != mon], df[df.mon == mon].copy()
    te["base"] = te.base_r4
    c = te["cov"].to_numpy(); X = te[c]
    b2 = te.base_r4.to_numpy().copy(); P["base"][mon] = b2
    for k, bcol in (("B", "base_r3"), ("C", "base_r4")):
        p = b2.copy(); p[c] += corr(tr[tr["cov"]], bcol, X, R10.FEAT_A, K2); P[k][mon] = p
    lc = te["lfpg"].to_numpy(); ltr = tr[tr["lfpg"]]
    for k, bcol in (("L0", "base_eq"), ("L1", "base_r4")):
        p = b2.copy()
        if lc.any() and len(ltr) >= 200:
            p[lc] += corr(ltr, bcol, te[lc], R9.FEAT_1, R9.PARAMS)
        P[k][mon] = p
    print(f"  month {mon}: base {rm(te.y - b2):.3f} | B {rm(te.y - P['B'][mon]):.3f} | C {rm(te.y - P['C'][mon]):.3f} | "
          f"L0 {rm(te.y - P['L0'][mon]):.3f} | L1 {rm(te.y - P['L1'][mon]):.3f}", flush=True)
y = {m: df[df.mon == m].y.to_numpy() for m in months}


def verdict(name, new, ref, need):
    per = {m: rm(y[m] - P[new][m]) - rm(y[m] - P[ref][m]) for m in months}
    changed = [m for m in months if np.any(P[new][m] != P[ref][m])]
    wins = sum(per[m] < 0 for m in changed)
    g = np.concatenate([(y[m] - P[ref][m]) ** 2 - (y[m] - P[new][m]) ** 2 for m in months]); s0 = np.concatenate([(y[m] - P[ref][m]) ** 2 for m in months])
    keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
    pooled = (float(np.sqrt(s0.mean())), float(np.sqrt((s0 - g).mean())))
    drop5 = (float(np.sqrt(s0[keep].mean())), float(np.sqrt((s0 - g)[keep].mean())))
    ok = pooled[1] < pooled[0] and wins >= need(len(changed)) and drop5[1] < drop5[0]
    print(f"{name}: pooled {pooled[0]:.3f} -> {pooled[1]:.3f} ({pooled[1] - pooled[0]:+.3f}) | wins {wins}/{len(changed)} | "
          f"best-5 removed {drop5[0]:.3f} -> {drop5[1]:.3f} => {'PASS' if ok else 'FAIL'}", flush=True)
    return {"pooled": pooled, "per_month": per, "wins": wins, "drop5": drop5, "PASS": bool(ok)}


out = {"stage2_C_vs_B": verdict("stage 2: C vs B", "C", "B", lambda n: 5), "lfpg_L1_vs_L0": verdict("LFPG: L1 vs L0", "L1", "L0", lambda n: 2 / 3 * n)}
(HERE / "round13.json").write_text(json.dumps(out, indent=2, default=float))
print("ALL DONE")
