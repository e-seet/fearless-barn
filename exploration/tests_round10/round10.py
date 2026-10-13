#!/usr/bin/env python3
"""tests_round10/PREREGISTERED.md: A (v48 design) vs K (capacity), F (+CatBoost family), M (+member disagreement).
Leave-one-month-out over Jan, Jul, Aug, Sep, Oct, Nov, Dec 2025."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_adsb_full"))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((ROOT / "tests_adsb_full" / "model_v48" / "config.json").read_text())
A7, FEAT_A, FEAT_ADSB, PARAMS, SEEDS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"], cfg["seeds"]
PARAMS_K = {**PARAMS, "n_estimators": 1500, "num_leaves": 63, "learning_rate": 0.02, "min_data_in_leaf": 50}
FEAT_M = FEAT_A + ["d_tree", "d_mlp", "d_ftt"]
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))


def frame():
    raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
    ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
    ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
    SURF = [("seasonal_full", "test", np.load(CACHE / "tree_seasonal_full_test.npy"), np.load(CACHE / "mlp6_seasonal_full_test.npy"), ftt_sf, {"2025-01", "2025-07"}),
            ("forward_inner", "val", np.load(CACHE / "tree_forward_inner_val.npy"), np.load(CACHE / "mlp18_forward_inner_val.npy"), ftt_fi, {"2025-08"}),
            ("forward", "val", np.load(CACHE / "tree_forward_val.npy"), np.load(CACHE / "mlp12_forward_val.npy"), z["pv"], {"2025-09", "2025-10"}),
            ("forward", "test", np.load(CACHE / "tree_forward_test.npy"), np.load(CACHE / "mlp12_forward_test.npy"), z["pt"], {"2025-11", "2025-12"})]
    frames = []
    for split, part, tree, mlp, ftt, prefixes in SURF:
        X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
        t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
        base = (tree + mlp + ftt) / 3
        df = pd.DataFrame({"y": np.asarray(y, float), "base": base, "d_tree": tree - base, "d_mlp": mlp - base, "d_ftt": ftt - base,
                           "ap": np.asarray(meta["ADEP_mvt"]).astype(str), "day": np.asarray(t.dt.strftime("%Y-%m-%d")),
                           "hour": np.asarray(t.dt.hour)}, index=X.index)
        days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in prefixes)
        e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
        frames.append(df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB]))
    df = pd.concat(frames); df = df[~df.index.duplicated()]
    df["mon"] = df.day.str[5:7]
    df["cov"] = df.e_first.notna() & df.ap.isin(A7)
    df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
    return df


def lgb_corr(trc, X, feat, params):
    return np.mean([lgb.LGBMRegressor(**{**params, "random_state": s}).fit(trc[feat], trc.y - trc.base).predict(X[feat]) for s in SEEDS], axis=0)


def cat_corr(trc, X, feat):
    from catboost import CatBoostRegressor
    def prep(d):
        d = d[feat].copy(); d["apc"] = d["apc"].astype(str); return d
    out = []
    for s in SEEDS:
        kw = dict(depth=8, iterations=2000, learning_rate=0.05, loss_function="RMSE", random_seed=s, verbose=False, cat_features=["apc"])
        try:
            m = CatBoostRegressor(task_type="GPU", **kw).fit(prep(trc), trc.y - trc.base)
        except Exception:
            m = CatBoostRegressor(thread_count=-1, **kw).fit(prep(trc), trc.y - trc.base)
        out.append(m.predict(prep(X)))
    return np.mean(out, axis=0)


def shuffled(part, seed):
    part = part.copy(); rng = np.random.default_rng(seed); idx = np.flatnonzero(part["cov"].to_numpy())
    cols = ["d_tree", "d_mlp", "d_ftt"]
    part.iloc[idx, [part.columns.get_loc(c) for c in cols]] = part.iloc[rng.permutation(idx)][cols].to_numpy()
    return part


if __name__ == "__main__":
    df = frame()
    months = ["01", "07", "08", "09", "10", "11", "12"]
    P = {k: {} for k in ("A", "K", "F", "M", "M_perm", "KF", "KM", "FM", "KFM")}
    for mon in months:
        tr, te = df[(df.mon != mon) & df["cov"]], df[df.mon == mon]
        c = te["cov"].to_numpy(); X = te[c]
        a = lgb_corr(tr, X, FEAT_A, PARAMS); k = lgb_corr(tr, X, FEAT_A, PARAMS_K)
        cb = cat_corr(tr, X, FEAT_A); m = lgb_corr(tr, X, FEAT_M, PARAMS)
        mp = lgb_corr(shuffled(tr, 11), shuffled(te, 12)[c], FEAT_M, PARAMS)
        km = lgb_corr(tr, X, FEAT_M, PARAMS_K)
        corr = {"A": a, "K": k, "F": (a + cb) / 2, "M": m, "M_perm": mp, "KF": (k + cb) / 2, "KM": km, "FM": (m + cb) / 2, "KFM": (km + cb) / 2}
        for key, v in corr.items():
            p = te.base.to_numpy().copy(); p[c] += v; P[key][mon] = p
        print(f"  month {mon}: " + " | ".join(f"{key} {rm(te.y - P[key][mon]):.3f}" for key in P), flush=True)
    y = {m: df[df.mon == m].y.to_numpy() for m in months}
    pooled = {k: float(np.sqrt(np.concatenate([(y[m] - P[k][m]) ** 2 for m in months]).mean())) for k in P}
    out = {"pooled": pooled, "tests": {}}
    for t in ("K", "F", "M"):
        per = {m: rm(y[m] - P[t][m]) - rm(y[m] - P["A"][m]) for m in months}
        wins = sum(v < 0 for v in per.values())
        g = np.concatenate([(y[m] - P["A"][m]) ** 2 - (y[m] - P[t][m]) ** 2 for m in months]); sA = np.concatenate([(y[m] - P["A"][m]) ** 2 for m in months])
        keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
        drop5 = (float(np.sqrt(sA[keep].mean())), float(np.sqrt((sA - g)[keep].mean())))
        ok = pooled[t] < pooled["A"] and wins >= 5 and drop5[1] < drop5[0] and (t != "M" or pooled["M"] < pooled["M_perm"])
        out["tests"][t] = {"delta": pooled[t] - pooled["A"], "wins": wins, "per_month": per, "drop5": drop5, "PASS_local": bool(ok)}
        print(f"{t}: pooled {pooled[t] - pooled['A']:+.3f} | {wins}/7 months | best-5 removed {drop5[0]:.3f}->{drop5[1]:.3f}"
              + (f" | perm {pooled['M_perm'] - pooled['A']:+.3f}" if t == "M" else "") + f" => {'PASS (local bars)' if ok else 'FAIL'}")
    print("POOLED:", {k: round(v - pooled["A"], 3) for k, v in pooled.items()}, "(delta vs A)")
    (HERE / "round10.json").write_text(json.dumps(out, indent=2, default=float))
