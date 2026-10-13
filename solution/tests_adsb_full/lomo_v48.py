#!/usr/bin/env python3
"""PREREGISTERED_v48.md: leave-one-month-out over 7 held-out months.
A = v46 corrector design; T = A + traffic features on all rows at the 7 airports; T_perm; S = A averaged over
seeds 42-46; drift gate for the traffic features (2025 vs 2026, Jan and Jul)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features, traffic
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((HERE / "model_v46" / "config.json").read_text())
A7, FEAT_A, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
TR = traffic.TRAFFIC
FEAT_T = FEAT_A + TR
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
SURF = [("seasonal_full", "test", (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf) / 3, {"2025-01", "2025-07"}),
        ("forward_inner", "val", (np.load(CACHE / "tree_forward_inner_val.npy") + np.load(CACHE / "mlp18_forward_inner_val.npy") + ftt_fi) / 3, {"2025-08"}),
        ("forward", "val", (np.load(CACHE / "tree_forward_val.npy") + np.load(CACHE / "mlp12_forward_val.npy") + z["pv"]) / 3, {"2025-09", "2025-10"}),
        ("forward", "test", (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]) / 3, {"2025-11", "2025-12"})]
trf = traffic.load()

# ---------------- drift gate (first: it can fail T on its own) ----------------
rk = cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
rk = rk[rk.PHASE_mvt.eq("DEP")]
r26 = trf.reindex(rk.MVT_ID_mvt.astype("int64")); r26["mon"] = pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
tr25 = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]); tr25 = tr25[tr25.PHASE_mvt.eq("DEP")]
r25 = trf.reindex(tr25.MVT_ID_mvt.astype("int64")); r25["mon"] = pd.to_datetime(tr25.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
drift = {}
for mon in (1, 7):
    a = r25[(r25.mon == mon) & r25.ap.isin(A7)]; b = r26[(r26.mon == mon) & r26.ap.isin(A7)]
    for c in TR:
        x, y = a[c].dropna(), b[c].dropna()
        sd = np.sqrt((x.var() + y.var()) / 2)
        drift[f"{c}_m{mon}"] = float((y.mean() - x.mean()) / sd) if sd > 0 else 0.0
drift_ok = all(abs(v) <= 0.2 for v in drift.values())
print("drift (SMD 2026 vs 2025, 7 airports):")
for c in TR:
    print(f"   {c:14s} Jan {drift[c + '_m1']:+.2f}  Jul {drift[c + '_m7']:+.2f}{'  DRIFT' if max(abs(drift[c + '_m1']), abs(drift[c + '_m7'])) > 0.2 else ''}")

# ---------------- LOMO frame ----------------
frames = []
for split, part, base, prefixes in SURF:
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
                       "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour)}, index=X.index)
    days = sorted(f.stem for f in (HERE / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB])
    frames.append(df)
df = pd.concat(frames); df = df[~df.index.duplicated()]
df = df.join(trf[TR], how="left")
df["mon"] = df.day.str[5:7]
df["covA"] = df.e_first.notna() & df.ap.isin(A7)
df["covT"] = df.ap.isin(A7)
df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
print(f"\n{len(df)} rows; A corrects {int(df.covA.sum())}, T corrects {int(df.covT.sum())}")


def shuffle_tr(part, seed):
    part = part.copy(); rng = np.random.default_rng(seed); m = part.covT.to_numpy()
    for a in A7:
        idx = np.flatnonzero(m & (part.ap.to_numpy() == a))
        part.iloc[idx, [part.columns.get_loc(c) for c in TR]] = part.iloc[rng.permutation(idx)][TR].to_numpy()
    return part


def run(tr, te, feat, cov, seeds=(42,)):
    trc = tr[tr[cov]]; c = te[cov].to_numpy()
    p = te.base.to_numpy().copy(); corr = np.zeros(c.sum())
    for s in seeds:
        m = lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(trc[feat], trc.y - trc.base)
        corr += m.predict(te.loc[c, feat]) / len(seeds)
    p[c] += corr
    return p


res, se = {}, {k: [] for k in ("base", "A", "T", "T_perm", "S")}
for mon in sorted(df.mon.unique()):
    tr, te = df[df.mon != mon], df[df.mon == mon]
    P = {"base": te.base.to_numpy(), "A": run(tr, te, FEAT_A, "covA"), "T": run(tr, te, FEAT_T, "covT"),
         "T_perm": run(shuffle_tr(tr, 1), shuffle_tr(te, 2), FEAT_T, "covT"),
         "S": run(tr, te, FEAT_A, "covA", seeds=(42, 43, 44, 45, 46))}
    res[mon] = {k: rm(te.y - v) for k, v in P.items()}
    covered, uncovered = te.covA.to_numpy(), (te.covT & ~te.covA).to_numpy()
    res[mon]["T_on_uncovered"] = (rm((te.y - P["base"])[uncovered]), rm((te.y - P["T"])[uncovered]), int(uncovered.sum()))
    for k, v in P.items():
        se[k].append(((te.y - v) ** 2).to_numpy())
    r = res[mon]
    print(f"  month {mon}: base {r['base']:.2f} | A {r['A']:.2f} | T {r['T']:.2f} ({r['T'] - r['A']:+.2f}) | T_perm {r['T_perm']:.2f} | "
          f"S {r['S']:.2f} ({r['S'] - r['A']:+.2f}) | T on rows without own track {r['T_on_uncovered'][0]:.1f} -> {r['T_on_uncovered'][1]:.1f} (n={r['T_on_uncovered'][2]})", flush=True)
pooled = {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in se.items()}
wT = sum(r["T"] < r["A"] for r in res.values()); wS = sum(r["S"] < r["A"] for r in res.values())
T_pass = pooled["T"] < pooled["A"] and wT >= 5 and pooled["T"] < pooled["T_perm"] and drift_ok
S_pass = pooled["S"] < pooled["A"] and wS >= 5
print(f"\nPOOLED: base {pooled['base']:.2f} | A {pooled['A']:.2f} | T {pooled['T']:.2f} ({pooled['T'] - pooled['A']:+.2f}) | "
      f"T_perm {pooled['T_perm']:.2f} | S {pooled['S']:.2f} ({pooled['S'] - pooled['A']:+.2f})")
print(f"T: pooled<A {pooled['T'] < pooled['A']} | {wT}/7 months | <T_perm {pooled['T'] < pooled['T_perm']} | drift ok {drift_ok} => {'PASS' if T_pass else 'FAIL'}")
print(f"S: pooled<A {pooled['S'] < pooled['A']} | {wS}/7 months => {'PASS' if S_pass else 'FAIL'}")
out = {"months": res, "pooled": pooled, "drift": drift, "T_PASS": bool(T_pass), "S_PASS": bool(S_pass)}
if T_pass and S_pass:
    se_ts = []
    for mon in sorted(df.mon.unique()):
        tr, te = df[df.mon != mon], df[df.mon == mon]
        se_ts.append(((te.y - run(tr, te, FEAT_T, "covT", seeds=(42, 43, 44, 45, 46))) ** 2).to_numpy())
    ts = float(np.sqrt(np.concatenate(se_ts).mean()))
    out["TS"] = ts; out["TS_beats_T"] = bool(ts < pooled["T"])
    print(f"TS (T with 5 seeds): {ts:.2f} vs T {pooled['T']:.2f} => {'use TS' if ts < pooled['T'] else 'use T'}")
(HERE / "lomo_v48.json").write_text(json.dumps(out, indent=2, default=float))
