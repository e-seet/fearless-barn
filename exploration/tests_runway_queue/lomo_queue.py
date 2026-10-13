#!/usr/bin/env python3
"""PREREGISTERED.md Strategy 1: leave-one-month-out. A+ = base + v46-design ADS-B corrector on its covered rows;
candidate = A+ plus corrector R (queue features + base + airport + hour) on every other row; R_perm control;
drift gate; best-5-rows-removed check."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_adsb_full")); sys.path.insert(0, str(HERE))
import features, queue_features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((ROOT / "tests_adsb_full" / "model_v46" / "config.json").read_text())
A7, FEAT_A, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
QF = queue_features.QF
ALL10 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
FEAT_R = QF + ["base", "apc10", "hour"]
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
q = queue_features.load()

# ---------------- drift gate on R's population ----------------
rk = cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]); rk = rk[rk.PHASE_mvt.eq("DEP")]
t25 = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]); t25 = t25[t25.PHASE_mvt.eq("DEP")]
drift = {}
for mon in (1, 7):
    a = q.reindex(t25.MVT_ID_mvt.astype("int64")[pd.to_datetime(t25.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy() == mon])
    b = q.reindex(rk.MVT_ID_mvt.astype("int64")[pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy() == mon])
    for c in QF:
        x, y = a[c].dropna(), b[c].dropna(); sd = np.sqrt((x.var() + y.var()) / 2)
        drift[f"{c}_m{mon}"] = float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0
drift_ok = all(abs(v) <= 0.2 for v in drift.values())
print("drift (SMD 2026 vs 2025):", {k: round(v, 2) for k, v in drift.items()}, "=> ok" if drift_ok else "=> DRIFT")

# ---------------- LOMO frame ----------------
z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
SURF = [("seasonal_full", "test", (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf) / 3, {"2025-01", "2025-07"}),
        ("forward_inner", "val", (np.load(CACHE / "tree_forward_inner_val.npy") + np.load(CACHE / "mlp18_forward_inner_val.npy") + ftt_fi) / 3, {"2025-08"}),
        ("forward", "val", (np.load(CACHE / "tree_forward_val.npy") + np.load(CACHE / "mlp12_forward_val.npy") + z["pv"]) / 3, {"2025-09", "2025-10"}),
        ("forward", "test", (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]) / 3, {"2025-11", "2025-12"})]
frames = []
for split, part, base, prefixes in SURF:
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
                       "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour)}, index=X.index)
    days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB])
    frames.append(df)
df = pd.concat(frames); df = df[~df.index.duplicated()]
df.index = df.index.astype("int64")
df = df.join(q[QF])
df["mon"] = df.day.str[5:7]
df["covA"] = df.e_first.notna() & df.ap.isin(A7)
df["covR"] = ~df.covA
df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
df["apc10"] = pd.Categorical(df.ap, categories=ALL10)
print(f"{len(df)} rows; A corrects {int(df.covA.sum())}, R corrects {int(df.covR.sum())}")


def shuffle_q(part, seed):
    part = part.copy(); rng = np.random.default_rng(seed); m = part.covR.to_numpy()
    for a in ALL10:
        idx = np.flatnonzero(m & (part.ap.to_numpy() == a))
        part.iloc[idx, [part.columns.get_loc(c) for c in QF]] = part.iloc[rng.permutation(idx)][QF].to_numpy()
    return part


def corr(tr, te, feat, cov):
    trc = tr[tr[cov]]; c = te[cov].to_numpy(); out = np.zeros(len(te))
    if c.any():
        m = lgb.LGBMRegressor(**PARAMS).fit(trc[feat], trc.y - trc.base)
        out[c] = m.predict(te.loc[c, feat])
    return out


res, se = {}, {k: [] for k in ("base", "A+", "cand", "cand_perm")}
gain_all, ap_all = [], []
for mon in sorted(df.mon.unique()):
    tr, te = df[df.mon != mon], df[df.mon == mon]
    pA = te.base.to_numpy() + corr(tr, te, FEAT_A, "covA")
    pC = pA + corr(tr, te, FEAT_R, "covR")
    pP = pA + corr(shuffle_q(tr, 1), shuffle_q(te, 2), FEAT_R, "covR")
    P = {"base": te.base.to_numpy(), "A+": pA, "cand": pC, "cand_perm": pP}
    res[mon] = {k: rm(te.y - v) for k, v in P.items()}
    for k, v in P.items():
        se[k].append(((te.y - v) ** 2).to_numpy())
    gain_all.append(((te.y - pA) ** 2 - (te.y - pC) ** 2).to_numpy()); ap_all.append(te.ap.to_numpy())
    r = res[mon]
    print(f"  month {mon}: base {r['base']:.2f} | A+ {r['A+']:.2f} | cand {r['cand']:.2f} ({r['cand'] - r['A+']:+.2f}) | cand_perm {r['cand_perm']:.2f}", flush=True)
pooled = {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in se.items()}
g = np.concatenate(gain_all); apa = np.concatenate(ap_all); seA = np.concatenate(se["A+"])
keep = np.ones(len(g), bool); keep[np.argsort(-g)[:5]] = False
drop5 = (float(np.sqrt(seA[keep].mean())), float(np.sqrt((seA - g)[keep].mean())))
per_ap = {a: (float(np.sqrt(seA[apa == a].mean())), float(np.sqrt((seA - g)[apa == a].mean()))) for a in ALL10}
wins = sum(r["cand"] < r["A+"] for r in res.values())
c1, c2, c3, c5 = pooled["cand"] < pooled["A+"], wins >= 5, pooled["cand"] < pooled["cand_perm"], drop5[1] < drop5[0]
print(f"\nPOOLED: base {pooled['base']:.2f} | A+ {pooled['A+']:.2f} | cand {pooled['cand']:.2f} ({pooled['cand'] - pooled['A+']:+.2f}) | cand_perm {pooled['cand_perm']:.2f}")
print("per airport A+ -> cand:", {a: f"{u:.1f}->{v:.1f}" for a, (u, v) in per_ap.items()})
print(f"best 5 rows removed: {drop5[0]:.2f} -> {drop5[1]:.2f}")
ok = c1 and c2 and c3 and drift_ok and c5
print(f"PRE-REGISTERED: pooled {c1} | {wins}/7 months {c2} | beats perm {c3} | drift ok {drift_ok} | best-5 removed {c5} => {'PASS' if ok else 'FAIL'}")
(HERE / "lomo_queue.json").write_text(json.dumps({"months": res, "pooled": pooled, "per_airport": per_ap, "drift": drift,
                                                   "drop5": drop5, "PASS": bool(ok)}, indent=2))
