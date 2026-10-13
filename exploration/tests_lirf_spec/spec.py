#!/usr/bin/env python3
"""tests_lirf_spec/PREREGISTERED.md: LIRF no-record specialist vs F (both LOBT-clipped), leave one month out."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostRegressor
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_copymix"))
from common import dataset as D
from common import io as cio
from copymix import times, lobt_clip
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
P = dict(depth=5, iterations=1000, learning_rate=0.04, l2_leaf_reg=10, loss_function="RMSE", verbose=0, thread_count=-1)
SEEDS = (0, 1, 2, 3, 4)
MONTHS = ["01", "07", "08", "09", "10", "11", "12"]


def fit_predict(Xtr, rtr, Xte, cats):
    return np.mean([CatBoostRegressor(**P, random_seed=s, cat_features=cats).fit(Xtr, rtr).predict(Xte) for s in SEEDS], axis=0)


def prep(X):
    X = X.copy(); cats = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cats: X[c] = X[c].astype(str).fillna("NA")
    return X, cats


if __name__ == "__main__":
    raw = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "FLIGHT_ID_mvt"])
    raw = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    X, y, _ = D.load_part("full", "train", "with_aobt", filter_target=False)
    y = pd.Series(np.asarray(y, float), index=X.index)
    ap = raw.ADEP_mvt.reindex(X.index).astype(str).to_numpy(); nm = raw.FLIGHT_ID_mvt.reindex(X.index).isna().to_numpy()
    mon = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True).dt.strftime("%m").to_numpy()
    mv, sc, lb = times(X.index, raw); g = np.maximum(mv - sc, 0)
    scope = (ap == "LIRF") & nm & np.isfinite(y.to_numpy())
    Xs, cats = prep(X[scope]); ys = y[scope].to_numpy(); gs = g[scope]; ms_ = mon[scope]
    print(f"scope rows 2025: {scope.sum()}", flush=True)
    S = pd.Series(np.nan, index=Xs.index)
    for m in MONTHS:
        tr, te = ms_ != m, ms_ == m
        r = fit_predict(Xs[tr], ys[tr] - gs[tr], Xs[te], cats)
        S[Xs.index[te]] = np.minimum(np.maximum(gs[te] + r, 30), np.maximum(gs[te], 88500))
        print(f"  S month {m}: rows {te.sum()} | S {rm(ys[te] - S[Xs.index[te]].to_numpy()):.0f} | identity {rm(ys[te] - gs[te]):.0f}", flush=True)
    S.to_frame("S").to_parquet(HERE / "S_heldout.parquet")
    d = pd.read_parquet(ROOT / "tests_copymix" / "cache" / "F_heldout.parquet"); d["y"] = y.reindex(d.index).to_numpy(); d = d[np.isfinite(d.y)]
    mvd, scd, lbd = times(d.index, raw)
    d["Fc"] = lobt_clip(d.F.to_numpy(), mvd, lbd)
    d["S"] = S.reindex(d.index).to_numpy()
    C = np.where(np.isfinite(d.S), d.S, d.Fc); d["C"] = lobt_clip(C, mvd, lbd)
    res = {}; sF, sC, jj = [], [], []
    for m in MONTHS:
        t = d[d.mon == m]; k = np.isfinite(t.S)
        res[m] = {"F": rm(t.y - t.Fc), "C": rm(t.y - t.C), "scope_F": rm((t.y - t.Fc)[k]), "scope_C": rm((t.y - t.C)[k]), "rows": int(k.sum())}
        sF.append(((t.y - t.Fc) ** 2).to_numpy()); sC.append(((t.y - t.C) ** 2).to_numpy()); jj.append(np.full(len(t), m in ("01", "07")))
        print(f"  month {m}: scope {res[m]['rows']} rows {res[m]['scope_F']:.0f} -> {res[m]['scope_C']:.0f} | month {res[m]['F']:.3f} -> {res[m]['C']:.3f} ({res[m]['C'] - res[m]['F']:+.3f})", flush=True)
    A, B, J = np.concatenate(sF), np.concatenate(sC), np.concatenate(jj)
    g_ = A - B; o = np.argsort(-g_)
    pooled = (float(np.sqrt(A.mean())), float(np.sqrt(B.mean()))); jul = (float(np.sqrt(A[J].mean())), float(np.sqrt(B[J].mean())))
    dk = {}
    for k in (5, 20, 50):
        keep = np.ones(len(g_), bool); keep[o[:k]] = False; dk[k] = (float(np.sqrt(A[keep].mean())), float(np.sqrt(B[keep].mean())))
    wins = sum(r["C"] < r["F"] for r in res.values())
    ok = pooled[1] < pooled[0] and jul[1] < jul[0] and wins >= 5 and dk[20][1] < dk[20][0]
    print(f"\nPOOLED {pooled[0]:.3f} -> {pooled[1]:.3f} ({pooled[1] - pooled[0]:+.3f}) | Jan+Jul {jul[0]:.3f} -> {jul[1]:.3f} ({jul[1] - jul[0]:+.3f}) | {wins}/7 | "
          + " | ".join(f"drop{k} {v[0]:.3f}->{v[1]:.3f}" for k, v in dk.items()) + f" => {'PASS' if ok else 'FAIL'}", flush=True)
    (HERE / "spec.json").write_text(json.dumps({"months": res, "pooled": pooled, "janjul": jul, "drop": dk, "wins": wins, "PASS": ok}, indent=2, default=float))
    print("ALL DONE")
