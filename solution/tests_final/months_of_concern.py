#!/usr/bin/env python3
"""tests_final/PREREGISTERED.md: the shipped pipeline and its month-scoped choices, re-checked on held-out Feb-Jun 2025.

Base: held-out holdout_febjun members (tree, MLP x6, small FTT x3) weighted per airport like v61's training base (c3d6).
Stages use the exact saved models (model_v61, model_v55, model_v56): none of them saw Feb-Jun 2025."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for p in (ROOT, ROOT / "tests_adsb_full", ROOT / "tests_copymix", ROOT / "tests_lirf_spec"):
    sys.path.insert(0, str(p))
import features
import final_pipeline as FP
from common import dataset as D
from common import io as cio
from copymix import times, lobt_clip
from spec import fit_predict, prep
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
FJ = ["02", "03", "04", "05", "06"]
W3 = json.loads((ROOT / "tests_ftt" / "final_weights_c3d6.json").read_text())["weights_per_airport"]

raw = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "FLIGHT_ID_mvt"])
raw = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
P = ROOT / "tests_febjun" / "preds"
tree = np.load(P / "tree_holdout_febjun_test.npy")
mlp = np.mean([np.load(P / f"mlp_holdout_febjun_test_seed{s}.npy") for s in range(42, 48)], axis=0)
ftt = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_holdout_febjun_lr1e-3.npz")["pt"] for s in (42, 43, 44)], axis=0)
X, y, meta = D.load_part("holdout_febjun", "test", "with_aobt", filter_target=False)
ap = np.asarray(meta["ADEP_mvt"]).astype(str)
w = np.array([W3.get(a, [1 / 3] * 3) for a in ap])
t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
df = pd.DataFrame({"y": np.asarray(y, float), "base": w[:, 0] * ftt + w[:, 1] * tree + w[:, 2] * mlp, "ap": ap,
                   "day": t.dt.strftime("%Y-%m-%d").to_numpy(), "hour": t.dt.hour.to_numpy(), "t": t.to_numpy()}, index=X.index)
df = df[np.isfinite(df.y)]
df["mon"] = df.day.str[5:7]
days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in {f"2025-{m}" for m in FJ})
e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
df = df.join(e[FP.ADSB].astype(float))
df["rec"] = raw.FLIGHT_ID_mvt.reindex(df.index).notna().to_numpy()
mv, sc, lb = times(df.index, raw); df["mv"], df["lb"] = mv, lb; df["g"] = np.maximum(mv - sc, 0)
print(f"Feb-Jun rows {len(df)} | ADS-B day files {len(days)} | base RMSE {rm(df.y - df.base):.2f}", flush=True)

# ---- stage 2: model_v61 on the 7 airports with a track
mc, bs = FP.boosters(ROOT / "tests_adsb_full" / "model_v61", "adsb_corrector")
cov = (df.e_first.notna() & df.ap.isin(mc["airports"])).to_numpy()
df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(mc["airports"])), categories=mc["airports"])
s2 = df.base.to_numpy().copy()
s2[cov] += np.mean([b.predict(df.loc[cov, mc["features"]]) for b in bs], axis=0)
s2 = np.clip(np.rint(s2), 0, 86400); no_adsb = ~cov | (np.rint(df.base.to_numpy()) == s2)

# ---- weather flags (same join as stage 3)
aps = df.ap.to_numpy(); tt = pd.to_datetime(df.t, utc=True)
raws = {st: FP.metar_raw(st, tt.min(), tt.max(), False) for st in sorted(set(aps))}
fw = FP.latest_metar(df.index.to_numpy(), tt.to_numpy(), aps, raws, {"tmpf": lambda d: FP.num(d["tmpf"]), "dwpf": lambda d: FP.num(d["dwpf"])}, ["tmpf", "dwpf"])
tw = FP.latest_metar(df.index.to_numpy(), tt.to_numpy(), aps, raws, {"is_ts": lambda d: d["wxcodes"].fillna("").str.upper().str.contains("TS")}, [])
frost = ((fw.tmpf <= 34.0) & ((fw.tmpf - fw.dwpf) <= 3.0)).to_numpy(); ts = tw.is_ts.fillna(False).to_numpy(dtype=bool)

# ---- stages 4-5 deltas
def stage(mdir, prefix, mask, pred):
    mc_, bs_ = FP.boosters(mdir, prefix); c = mask & df.rec.to_numpy() & df.e_first.notna().to_numpy()
    out = np.zeros(len(df))
    if c.any():
        out[c] = np.mean([b.predict(df.loc[c, mc_["features"]]) for b in bs_], axis=0)
    return out, c
dL, cL = stage(ROOT / "tests_lirf_adsb" / "model_v55", "lirf_corrector", aps == "LIRF", None)
dP, cP = stage(ROOT / "tests_round9" / "model_v56", "lfpg_corrector", aps == "LFPG", None)

# ---- LIRF no-record specialist, leave one month out over the 12 months (scope rows only)
Xall, yall, _ = D.load_part("full", "train", "with_aobt", filter_target=False)
yall = pd.Series(np.asarray(yall, float), index=Xall.index)
apa = raw.ADEP_mvt.reindex(Xall.index).astype(str).to_numpy(); nma = raw.FLIGHT_ID_mvt.reindex(Xall.index).isna().to_numpy()
mona = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(Xall.index), utc=True).dt.strftime("%m").to_numpy()
mva, sca, _ = times(Xall.index, raw); ga = np.maximum(mva - sca, 0)
scope = (apa == "LIRF") & nma & np.isfinite(yall.to_numpy())
Xs, cats = prep(Xall[scope]); ys = yall[scope].to_numpy(); gs = ga[scope]; ms_ = mona[scope]
S = pd.Series(np.nan, index=Xs.index)
for m in FJ:
    tr, te = ms_ != m, ms_ == m
    if te.any():
        r = fit_predict(Xs[tr], ys[tr] - gs[tr], Xs[te], cats)
        S[Xs.index[te]] = np.minimum(np.maximum(gs[te] + r, 30), np.maximum(gs[te], 88500))
        print(f"  specialist month {m}: {te.sum()} rows", flush=True)
S.to_frame("S").to_parquet(HERE / "S_febjun.parquet")
Sx = S.reindex(df.index).to_numpy()

# ---- arms (each a full prediction over Feb-Jun)
mon = df.mon.to_numpy()
shipped = s2.copy()                                     # stage 3 inactive in Feb-Jun as shipped; LIRF stage inactive outside Jan
shipped = np.clip(np.rint(shipped + dP), 60, 86400)
arms = {"base": df.base.to_numpy(), "stage2": s2, "shipped": shipped}
arms["+frost_any_month"] = np.clip(np.rint(shipped + 30.0 * (no_adsb & frost)), 60, 86400)
arms["+ts_any_month"] = np.clip(np.rint(shipped + 200.0 * (no_adsb & ts)), 60, 86400)
arms["+lirf_corr_any_month"] = np.clip(np.rint(shipped + dL), 60, 86400)
arms["+lirf_specialist"] = np.where(np.isfinite(Sx), Sx, shipped)
arms["+lirf_hybrid_6h"] = np.where(np.isfinite(Sx) & (df.g.to_numpy() > 21600), Sx, shipped)
arms["+lobt_clip"] = lobt_clip(shipped, df.mv.to_numpy(), df.lb.to_numpy())
res = {}
yv = df.y.to_numpy()
for k, p in arms.items():
    res[k] = {m: rm(yv[mon == m] - p[mon == m]) for m in FJ}
    res[k]["Feb+Jun"] = rm(yv[np.isin(mon, ["02", "06"])] - p[np.isin(mon, ["02", "06"])]); res[k]["Feb-Jun"] = rm(yv - p)
tab = pd.DataFrame(res).T.round(3); print(tab.to_string(), flush=True)

def robust(new, ref, k=20):
    g = (yv - arms[ref]) ** 2 - (yv - arms[new]) ** 2; o = np.argsort(-g); keep = np.ones(len(g), bool); keep[o[:k]] = False
    a, b = (yv - arms[ref]) ** 2, (yv - arms[new]) ** 2
    return {"rows_changed": int((arms[new] != arms[ref]).sum()), "delta": float(np.sqrt(b.mean()) - np.sqrt(a.mean())),
            f"delta_drop{k}": float(np.sqrt(b[keep].mean()) - np.sqrt(a[keep].mean())),
            "months_better": int(sum(res[new][m] < res[ref][m] for m in FJ if np.any(arms[new][mon == m] != arms[ref][mon == m]))),
            "months_changed": int(sum(np.any(arms[new][mon == m] != arms[ref][mon == m]) for m in FJ))}
checks = {k: robust(k, "shipped") for k in arms if k.startswith("+")}
checks["stage2_vs_base"] = robust("stage2", "base"); checks["shipped_vs_stage2"] = robust("shipped", "stage2")
for k, v in checks.items():
    print(f"{k}: {v}", flush=True)
(HERE / "months_of_concern.json").write_text(json.dumps({"rmse": res, "checks": checks,
    "coverage": {m: float(cov[mon == m].mean()) for m in FJ}, "lirf_rows": {m: int(cL[mon == m].sum()) for m in FJ},
    "lfpg_rows": {m: int(cP[mon == m].sum()) for m in FJ}}, indent=2, default=float))
print("ALL DONE")
