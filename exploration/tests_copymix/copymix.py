#!/usr/bin/env python3
"""tests_copymix/PREREGISTERED.md: copy-mixture member M blended per airport with the current recipe F; LOMO over 7 months.
Saves held-out F and M (cache/) so the build can reuse them."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_round10", ROOT / "tests_round13"):
    sys.path.insert(0, str(p))
from common import dataset as D
from common import io as cio
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
MONTHS = ["01", "07", "08", "09", "10", "11", "12"]
CLS = dict(objective="binary", learning_rate=0.05, n_estimators=500, num_leaves=127, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, random_state=42, verbose=-1)
REG = dict(objective="l2", learning_rate=0.05, n_estimators=800, num_leaves=255, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, random_state=42, verbose=-1)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def times(df_ids, src):
    s = src.reindex(df_ids)
    sec = lambda c: (pd.to_datetime(s[c], utc=True) - EPOCH).dt.total_seconds().to_numpy()
    return sec("MVT_TIME_UTC_mvt"), sec("SCHED_TIME_UTC_mvt"), sec("LOBT_flt")


def mixture(p, yn, mvt, sched, lobt):
    ms = np.clip(mvt - sched, 0, 86400)
    win_ok = np.abs(sched - lobt) <= 3606
    p = np.where(np.isfinite(lobt) & ~win_ok, 0.0, p)
    m = p * ms + (1 - p) * yn
    return lobt_clip(m, mvt, lobt)


def lobt_clip(pred, mvt, lobt):
    lw = mvt - lobt; ok = np.isfinite(lw)
    lo = np.where(ok, np.maximum(lw - 3606, 0), 0); hi = np.where(ok, lw + 3606, np.inf)
    return np.clip(pred, lo, hi)


if __name__ == "__main__":
    raw = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "BLOCK_TIME_UTC_mvt"])
    raw = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    X, y, meta = D.load_part("full", "train", "with_aobt", filter_target=False)
    y = pd.Series(np.asarray(y, float), index=X.index)
    mon = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True).dt.strftime("%m").to_numpy()
    blk = (pd.to_datetime(raw.BLOCK_TIME_UTC_mvt.reindex(X.index), utc=True) - pd.to_datetime(raw.SCHED_TIME_UTC_mvt.reindex(X.index), utc=True)).dt.total_seconds()
    copy = (blk.abs() <= 60).to_numpy()
    mvt, sched, lobt = times(X.index, raw)
    ok = np.isfinite(y.to_numpy())
    print(f"2025 DEP rows {len(X)}; copy rate {copy[ok].mean():.4f}", flush=True)

    # ---- held-out M for the 7 scored months
    fM = CACHE / "M_heldout.parquet"
    if not fM.exists():
        out = []
        for m in MONTHS:
            tr = (mon != m) & ok; te = mon == m
            cls = lgb.LGBMClassifier(**CLS).fit(X[tr], copy[tr])
            reg = lgb.LGBMRegressor(**REG).fit(X[tr & ~copy], np.clip(y[tr & ~copy], 0, 7200))
            p = cls.predict_proba(X[te])[:, 1]; yn = reg.predict(X[te])
            M = mixture(p, yn, mvt[te], sched[te], lobt[te])
            out.append(pd.DataFrame({"M": M, "p": p, "yn": yn}, index=X.index[te]))
            yy = y[te].to_numpy(); k = np.isfinite(yy)
            print(f"  M month {m}: RMSE {rm(yy[k] - M[k]):.2f} | mean p {p.mean():.4f} vs copy rate {copy[te & ok].mean():.4f}", flush=True)
        pd.concat(out).to_parquet(fM)
    Mdf = pd.read_parquet(fM)

    # ---- held-out F (round-13 frame: realistic-4 base + K2 corrector trained on the round-12 base)
    fF = CACHE / "F_heldout.parquet"
    if not fF.exists():
        import round10 as R10
        src = (ROOT / "tests_round13" / "round13.py").read_text(encoding="utf-8").split("def corr(")[0]
        ns = {"__file__": str(ROOT / "tests_round13" / "round13.py"), "__name__": "r13"}
        exec(compile(src, "r13", "exec"), ns)
        df = ns["df"]; K2 = ns["K2"]
        rows = []
        for m in MONTHS:
            tr, te = df[(df.mon != m) & df["cov"]].copy(), df[df.mon == m].copy()
            tr["base"] = tr.base_r3; te["base"] = te.base_r4
            c = te["cov"].to_numpy(); p = te.base_r4.to_numpy().copy()
            p[c] += np.mean([lgb.LGBMRegressor(**{**K2, "random_state": s}).fit(tr[R10.FEAT_A], tr.y - tr.base).predict(te.loc[c, R10.FEAT_A]) for s in R10.SEEDS], axis=0)
            rows.append(pd.DataFrame({"F": p, "ap": te.ap.to_numpy(), "mon": m}, index=te.index))
            print(f"  F month {m}: RMSE {rm(te.y - p):.3f}", flush=True)
        pd.concat(rows).to_parquet(fF)
    Fdf = pd.read_parquet(fF)

    d = Fdf.join(Mdf, how="inner"); d["y"] = y.reindex(d.index).to_numpy()
    d = d[np.isfinite(d.y)]
    mv, sc, lb = times(d.index, raw)
    res, se = {}, {"F": [], "C": []}; Wall = {}
    for m in MONTHS:
        trm, tem = d[d.mon != m], d[d.mon == m]
        W = {}
        for a, g in trm.groupby("ap"):
            dlt = (g.M - g.F).to_numpy(); r = (g.y - g.F).to_numpy()
            W[a] = float(np.clip((dlt @ r) / (dlt @ dlt), 0, 1)) if (dlt @ dlt) > 0 else 0.0
        w = tem.ap.map(W).fillna(0).to_numpy()
        k = (d.mon == m).to_numpy()
        C = lobt_clip((1 - w) * tem.F.to_numpy() + w * tem.M.to_numpy(), mv[k], lb[k])
        res[m] = {"F": rm(tem.y - tem.F), "C": rm(tem.y.to_numpy() - C), "weights": W}
        se["F"].append(((tem.y - tem.F) ** 2).to_numpy()); se["C"].append((tem.y.to_numpy() - C) ** 2)
        print(f"  month {m}: F {res[m]['F']:.3f} | combined {res[m]['C']:.3f} ({res[m]['C'] - res[m]['F']:+.3f}) | weights {({a: round(v, 2) for a, v in W.items()})}", flush=True)
    sF, sC = np.concatenate(se["F"]), np.concatenate(se["C"])
    yall = np.concatenate([d[d.mon == m].y.to_numpy() for m in MONTHS])
    g = sF - sC; keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
    body = yall <= 3600
    pooled = (float(np.sqrt(sF.mean())), float(np.sqrt(sC.mean())))
    drop5 = (float(np.sqrt(sF[keep].mean())), float(np.sqrt(sC[keep].mean())))
    wins = sum(r["C"] < r["F"] for r in res.values())
    share_body = float(g[body].sum() / g.sum()) if g.sum() != 0 else 0.0
    ok_ = pooled[1] < pooled[0] and wins >= 5 and drop5[1] < drop5[0]
    print(f"\nPOOLED: F {pooled[0]:.3f} -> combined {pooled[1]:.3f} ({pooled[1] - pooled[0]:+.3f}) | {wins}/7 months | best-5 removed "
          f"{drop5[0]:.3f} -> {drop5[1]:.3f} | share of the squared-error gain from body rows (y<=3600) {share_body:.2f} => {'PASS' if ok_ else 'FAIL'}")
    (HERE / "copymix.json").write_text(json.dumps({"months": res, "pooled": pooled, "drop5": drop5, "wins": wins, "body_share": share_body, "PASS": bool(ok_)}, indent=2))
    print("ALL DONE")
