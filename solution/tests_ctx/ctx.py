#!/usr/bin/env python3
"""tests_ctx/PREREGISTERED.md: context corrector C on every row, 7 months LOMO + held-out Feb-Jun 2025, drift gate."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_copymix"):
    sys.path.insert(0, str(p))
import features
import final_pipeline as FP
from common import dataset as D
from common import io as cio
from copymix import times, lobt_clip
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
PARAMS = dict(objective="l2", learning_rate=0.03, n_estimators=800, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1)
SEEDS = (42, 43, 44)
CTX = ["nb_proxy_20", "nb_proxy_60", "nb_proxy_rwy30", "nb_late_15", "nb_late_60", "own_late_rel"]
FEAT = CTX + ["F", "apc", "hour", "own_proxy", "own_late", "adsb"]
M7 = ["01", "07", "08", "09", "10", "11", "12"]; FJ = ["02", "03", "04", "05", "06"]


def sec(s):
    return (pd.to_datetime(s, utc=True) - EPOCH).dt.total_seconds().to_numpy()


def window_mean(t, v, lo, hi, group):
    """Mean of v over OTHER rows of the same group with time in [t+lo, t+hi) (hi=0: strictly before; symmetric windows include both sides)."""
    out = np.full(len(t), np.nan)
    for g in pd.unique(group):
        idx = np.flatnonzero(group == g)
        o = idx[np.argsort(t[idx], kind="stable")]
        tt = t[o]; vv = v[o]; ok = np.isfinite(vv)
        cs = np.concatenate([[0], np.cumsum(np.where(ok, vv, 0))]); cn = np.concatenate([[0], np.cumsum(ok)])
        a = np.searchsorted(tt, tt + lo, "left"); b = np.searchsorted(tt, tt + hi, "left" if hi <= 0 else "right")
        s = cs[b] - cs[a]; n = cn[b] - cn[a]
        if hi > 0:                       # symmetric window contains the row itself: remove it
            s = s - np.where(ok, vv, 0); n = n - ok
        out[o] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def context(mov):
    """Context features for the DEP rows of a movements frame (index MVT_ID_mvt)."""
    d = mov[mov.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    t = sec(d.MVT_TIME_UTC_mvt); ap = d.ADEP_mvt.astype(str).to_numpy()
    proxy = np.clip(t - sec(d.AOBT_3_flt), 0, 7200); late = np.clip(t - sec(d.SCHED_TIME_UTC_mvt), -3600, 14400)
    rwy = (d.ADEP_mvt.astype(str) + "|" + d.RUNWAY_mvt.astype(str)).to_numpy()
    f = pd.DataFrame(index=d.index)
    f["nb_proxy_20"] = window_mean(t, proxy, -1200, 0, ap)
    f["nb_proxy_60"] = window_mean(t, proxy, -3600, 0, ap)
    f["nb_proxy_rwy30"] = window_mean(t, proxy, -1800, 0, rwy)
    f["nb_late_15"] = window_mean(t, late, -900, 900, ap)
    f["nb_late_60"] = window_mean(t, late, -3600, 3600, ap)
    f["own_late_rel"] = late - f.nb_late_60.to_numpy()
    f["own_proxy"], f["own_late"] = proxy, late
    f["hour"] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.hour.to_numpy()
    return f


def fit_predict(tr, te):
    k = (tr.y <= 10800) & (tr.F <= 7200)
    return np.mean([lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(tr.loc[k, FEAT], (tr.y - tr.F)[k]).predict(te[FEAT]) for s in SEEDS], axis=0)


def score(name, y, ref, new, mon, months, pair, need):
    a, b = (y - ref) ** 2, (y - new) ** 2; g = a - b; o = np.argsort(-g)
    keep = np.ones(len(g), bool); keep[o[:20]] = False
    per = {m: float(np.sqrt(b[mon == m].mean()) - np.sqrt(a[mon == m].mean())) for m in months}
    pj = np.isin(mon, pair); body = y <= 3600
    r = {"pooled": (float(np.sqrt(a.mean())), float(np.sqrt(b.mean()))), "pair": (float(np.sqrt(a[pj].mean())), float(np.sqrt(b[pj].mean()))),
         "drop20": (float(np.sqrt(a[keep].mean())), float(np.sqrt(b[keep].mean()))), "months": per,
         "wins": int(sum(v < 0 for v in per.values())), "body_gain": float(g[body].sum()), "tail_gain": float(g[~body].sum())}
    r["PASS"] = r["pooled"][1] < r["pooled"][0] and r["pair"][1] < r["pair"][0] and r["wins"] >= need and r["drop20"][1] < r["drop20"][0] and r["body_gain"] > 0
    print(f"{name}: pooled {r['pooled'][0]:.3f} -> {r['pooled'][1]:.3f} ({r['pooled'][1]-r['pooled'][0]:+.3f}) | {'+'.join(pair)} {r['pair'][1]-r['pair'][0]:+.3f} | "
          f"{r['wins']}/{len(months)} | drop20 {r['drop20'][1]-r['drop20'][0]:+.3f} | body gain {r['body_gain']:.3g} tail gain {r['tail_gain']:.3g} | "
          f"months {({m: round(v, 3) for m, v in per.items()})} => {'PASS' if r['PASS'] else 'FAIL'}", flush=True)
    return r


if __name__ == "__main__":
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "RUNWAY_mvt"]
    tr_mov = cio.load_training(columns=cols)
    C25 = context(tr_mov); print("2025 context rows", len(C25), flush=True)
    raw = tr_mov[tr_mov.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    X, y, _ = D.load_part("full", "train", "with_aobt", filter_target=False); yall = pd.Series(np.asarray(y, float), index=X.index)

    # ---- drift gate (Jan + Jul): 2025 training vs final_ranking 2026
    fr = pd.read_parquet(ROOT.parent / "prc-2026-datasets" / "final_ranking.parquet", columns=cols)
    C26 = context(fr)
    m25 = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(C25.index), utc=True).dt.month.isin([1, 7]).to_numpy()
    r26 = fr[fr.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    m26 = pd.to_datetime(r26.MVT_TIME_UTC_mvt.reindex(C26.index), utc=True).dt.month.isin([1, 7]).to_numpy()
    drift = {c: float(abs(np.nanmean(C26[c].to_numpy()[m26]) - np.nanmean(C25[c].to_numpy()[m25])) / np.nanstd(C25[c].to_numpy()[m25])) for c in CTX + ["own_proxy", "own_late"]}
    drift_ok = all(v <= 0.2 for v in drift.values())
    print("drift (standardized mean shift, Jan+Jul 2026 vs 2025):", {k: round(v, 3) for k, v in drift.items()}, "=>", "PASS" if drift_ok else "FAIL", flush=True)

    # ---- surface 1: 7 months
    d = pd.read_parquet(ROOT / "tests_copymix" / "cache" / "F_heldout.parquet"); d["y"] = yall.reindex(d.index).to_numpy(); d = d[np.isfinite(d.y)]
    days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("2025-*.parquet") if f.stem[5:7] in M7)
    e = pd.concat([features.load_day(x) for x in days]); e = e[~e.index.duplicated()]
    A7 = json.loads((ROOT / "tests_adsb_full" / "model_v61" / "config.json").read_text())["airports"]
    d["adsb"] = (d.index.isin(e.index[e.e_first.notna()]) & d.ap.isin(A7)).astype(int)
    d = d.join(C25)
    mv, sc, lb = times(d.index, raw); d["Fc"] = lobt_clip(d.F.to_numpy(), mv, lb)
    APS = sorted(d.ap.unique()); d["apc"] = pd.Categorical(d.ap, categories=APS)
    new = d.Fc.to_numpy().copy()
    for m in M7:
        tr, te = d[d.mon != m], d[d.mon == m]
        k = (te.F <= 7200).to_numpy(); p = te.F.to_numpy().copy()
        p[k] += fit_predict(tr, te[k]); new[(d.mon == m).to_numpy()] = lobt_clip(p, mv[(d.mon == m).to_numpy()], lb[(d.mon == m).to_numpy()])
        print(f"  7m month {m} done", flush=True)
    r1 = score("7 months", d.y.to_numpy(), d.Fc.to_numpy(), new, d.mon.to_numpy(), M7, ["01", "07"], 5)

    # ---- surface 2: held-out Feb-Jun (shipped design), C trained on all 7 months
    W3 = json.loads((ROOT / "tests_ftt" / "final_weights_c3d6.json").read_text())["weights_per_airport"]
    P = ROOT / "tests_febjun" / "preds"
    tree = np.load(P / "tree_holdout_febjun_test.npy"); mlp = np.mean([np.load(P / f"mlp_holdout_febjun_test_seed{s}.npy") for s in range(42, 48)], axis=0)
    ftt = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_holdout_febjun_lr1e-3.npz")["pt"] for s in (42, 43, 44)], axis=0)
    Xf, yf, meta = D.load_part("holdout_febjun", "test", "with_aobt", filter_target=False)
    apf = np.asarray(meta["ADEP_mvt"]).astype(str); w = np.array([W3.get(a, [1 / 3] * 3) for a in apf])
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(Xf.index), utc=True)
    f = pd.DataFrame({"y": np.asarray(yf, float), "base": w[:, 0] * ftt + w[:, 1] * tree + w[:, 2] * mlp, "ap": apf,
                      "mon": t.dt.strftime("%m").to_numpy(), "hour": t.dt.hour.to_numpy()}, index=Xf.index)
    f = f[np.isfinite(f.y)]
    daysf = sorted(x.stem for x in (ROOT / "tests_adsb_full" / "points").glob("2025-*.parquet") if x.stem[5:7] in FJ)
    ef = pd.concat([features.load_day(x) for x in daysf]).sort_values("liftoff_dt"); ef = ef[~ef.index.duplicated()]
    f = f.join(ef[FP.ADSB].astype(float))
    mc, bs = FP.boosters(ROOT / "tests_adsb_full" / "model_v61", "adsb_corrector")
    cov = (f.e_first.notna() & f.ap.isin(mc["airports"])).to_numpy()
    f["apc"] = pd.Categorical(f.ap.where(f.ap.isin(mc["airports"])), categories=mc["airports"])
    s2 = f.base.to_numpy().copy(); s2[cov] += np.mean([b.predict(f.loc[cov, mc["features"]]) for b in bs], axis=0)
    s2 = np.clip(np.rint(s2), 0, 86400)
    fid = cio.load_training(columns=["MVT_ID_mvt", "FLIGHT_ID_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").FLIGHT_ID_mvt.reindex(f.index).notna().to_numpy()
    mcL, bsL = FP.boosters(ROOT / "tests_round9" / "model_v56", "lfpg_corrector")
    cL = (f.ap.eq("LFPG") & f.e_first.notna()).to_numpy() & fid
    if cL.any():
        s2[cL] = np.clip(np.rint(s2[cL] + np.mean([b.predict(f.loc[cL, mcL["features"]]) for b in bsL], axis=0)), 60, 86400)
    f["F"] = s2; f["adsb"] = cov.astype(int); f = f.drop(columns=["apc"]).join(C25.drop(columns=["hour"]))
    mvf, scf, lbf = times(f.index, raw); f["Fc"] = lobt_clip(f.F.to_numpy(), mvf, lbf)
    f["apc"] = pd.Categorical(f.ap, categories=APS)
    print(f"Feb-Jun shipped (clipped) RMSE {rm(f.y - f.Fc):.3f}", flush=True)
    k = (f.F <= 7200).to_numpy(); p = f.F.to_numpy().copy(); p[k] += fit_predict(d, f[k])
    newf = lobt_clip(p, mvf, lbf)
    r2 = score("Feb-Jun", f.y.to_numpy(), f.Fc.to_numpy(), newf, f.mon.to_numpy(), FJ, ["02", "06"], 4)

    ok = r1["PASS"] and r2["PASS"] and drift_ok
    print(f"\nVERDICT => {'PASS' if ok else 'FAIL'}", flush=True)
    (HERE / "ctx.json").write_text(json.dumps({"drift": drift, "drift_ok": drift_ok, "seven_months": r1, "febjun": r2, "PASS": ok}, indent=2, default=float))
    print("ALL DONE")
