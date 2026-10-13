#!/usr/bin/env python3
"""PREREGISTERED_v46.md: Aug 2025 fresh surface. Q1 +Sep-Oct training, Q2 capacity variant, Q3 LIRF."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((HERE / "model_v45" / "config.json").read_text())
A7, FEAT, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
BIG = {**PARAMS, "n_estimators": 600, "num_leaves": 31}
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")


def frame(split, part, base, prefixes):
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    assert len(base) == len(X), (split, part, len(base), len(X))
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
                       "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour)}, index=X.index)
    days = sorted(f.stem for f in (HERE / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB])
    df["has"] = df.e_first.notna()
    return df, days


def cat(df, aps):
    df = df.copy(); df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(aps)), categories=aps); return df


def fit(tr, aps, params=PARAMS, shuffle=None):
    tr = cat(tr[tr.has & tr.ap.isin(aps)], aps)
    if shuffle is not None:
        rng = np.random.default_rng(shuffle)
        for a in aps:
            idx = np.flatnonzero(tr.ap.to_numpy() == a)
            tr.iloc[idx, [tr.columns.get_loc(c) for c in FEAT_ADSB]] = tr.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    return lgb.LGBMRegressor(**params).fit(tr[FEAT], tr.y - tr.base), aps


def apply(model_aps, te, predict=None):
    m, aps = model_aps
    te = cat(te, aps); c = (te.has & te.ap.isin(aps)).to_numpy()
    p = te.base.to_numpy().copy()
    if c.any():
        p[c] += (predict or m.predict)(te.loc[c, FEAT])
    return p


def shuffled(te, aps, seed):
    te = te.copy(); rng = np.random.default_rng(seed); has = te.has.to_numpy()
    for a in aps:
        idx = np.flatnonzero(has & (te.ap.to_numpy() == a))
        te.iloc[idx, [te.columns.get_loc(c) for c in FEAT_ADSB]] = te.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    return te


# training surfaces (all held-out bases)
ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
jj, _ = frame("seasonal_full", "test", (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf) / 3,
              {"2025-01", "2025-07"})
z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
nd, _ = frame("forward", "test", (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]) / 3,
              {"2025-11", "2025-12"})
so, _ = frame("forward", "val", (np.load(CACHE / "tree_forward_val.npy") + np.load(CACHE / "mlp12_forward_val.npy") + z["pv"]) / 3,
              {"2025-09", "2025-10"})
ftt_fi = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_forward_inner_lr1e-3.npz")["pv"] for s in range(42, 51)], axis=0)
au, au_days = frame("forward_inner", "val", (np.load(CACHE / "tree_forward_inner_val.npy") + np.load(CACHE / "mlp18_forward_inner_val.npy") + ftt_fi) / 3,
                    {"2025-08"})
print(f"Aug: {len(au_days)} of 31 days, {len(au)} rows; covered by airport:", au.groupby("ap").has.mean().round(2).to_dict())
out = {"aug_days": len(au_days)}
b0 = rm(au.y - au.base)


def perday(p, mask=None):
    m = np.ones(len(au), bool) if mask is None else mask
    return {d: rm((au.y - p)[m & (au.day == d).to_numpy()]) - rm((au.y - au.base)[m & (au.day == d).to_numpy()]) for d in sorted(au.day.unique())}


# ---------------- Q1: R45 (saved) vs R46 (+Sep-Oct) ----------------
b45 = lgb.Booster(model_file=str(HERE / "model_v45" / "adsb_corrector.txt"))
p45 = apply((None, A7), au, predict=b45.predict)
tr45 = pd.concat([jj, nd]); tr46 = pd.concat([jj, nd, so])
p46 = apply(fit(tr46, A7), au)
d45, d46 = perday(p45), perday(p46)
w = sum(d46[d] < d45[d] for d in d45)
q1 = rm(au.y - p46) < rm(au.y - p45) and w >= len(d45) / 2
out["Q1"] = {"base": b0, "R45": rm(au.y - p45), "R46": rm(au.y - p46), "R46_day_wins": w, "n_days": len(d45),
             "R45_days_better_than_base": sum(v < 0 for v in d45.values()), "ADOPT_R46": bool(q1)}
print(f"\nQ1: base {b0:.2f} | R45 (saved v45 corrector) {rm(au.y - p45):.2f} ({rm(au.y - p45) - b0:+.2f}, "
      f"{out['Q1']['R45_days_better_than_base']}/{len(d45)} days better than base) | R46 (+Sep-Oct) {rm(au.y - p46):.2f} "
      f"(wins {w}/{len(d45)} vs R45) => {'ADOPT R46' if q1 else 'keep R45 training set'}")
print("   per airport base/R45/R46:", {a: (round(rm((au.y - au.base)[au.ap == a]), 1), round(rm((au.y - p45)[au.ap == a]), 1),
                                          round(rm((au.y - p46)[au.ap == a]), 1)) for a in A7})
train = tr46 if q1 else tr45
p_q1 = p46 if q1 else p45

# ---------------- Q2: capacity variant ----------------
pC = apply(fit(train, A7, BIG), au)
dq, dC = perday(p_q1), perday(pC)
wC = sum(dC[d] < dq[d] for d in dq)
q2 = rm(au.y - pC) < rm(au.y - p_q1) and wC >= len(dq) / 2
out["Q2"] = {"q1_winner": rm(au.y - p_q1), "C": rm(au.y - pC), "C_day_wins": wC, "ADOPT_C": bool(q2)}
print(f"Q2: Q1 winner {rm(au.y - p_q1):.2f} | capacity variant {rm(au.y - pC):.2f} (wins {wC}/{len(dq)}) => {'ADOPT' if q2 else 'keep'}")
params = BIG if q2 else PARAMS
p_q2 = pC if q2 else p_q1

# ---------------- Q3: LIRF ----------------
s = (au.ap == "LIRF").to_numpy()
pL = apply(fit(train, ["LIRF"], params), au)
br, nr = rm((au.y - au.base)[s]), rm((au.y - pL)[s])
pdL = perday(pL, s)
perms = [rm((au.y - apply(fit(train, ["LIRF"], params, shuffle=k), shuffled(au, ["LIRF"], k)))[s]) for k in range(5)]
g = ((au.y - au.base) ** 2 - (au.y - pL) ** 2).to_numpy()
keep = s.copy(); keep[np.argsort(-np.where(s, g, -np.inf))[:5]] = False
drop5 = (rm((au.y - au.base)[keep]), rm((au.y - pL)[keep]))
q3 = nr < br and sum(v < 0 for v in pdL.values()) >= 2 / 3 * len(pdL) and nr < min(perms) and drop5[1] < drop5[0]
out["Q3"] = {"lirf_base": br, "lirf_new": nr, "days_better": sum(v < 0 for v in pdL.values()), "perm_best": min(perms),
             "drop5": drop5, "covered": float(au.has[s].mean()), "ADD": bool(q3)}
print(f"Q3 LIRF: {br:.1f} -> {nr:.1f}, days better {out['Q3']['days_better']}/{len(pdL)}, permuted best {min(perms):.1f}, "
      f"5 best rows removed {drop5[0]:.1f} -> {drop5[1]:.1f}, covered {au.has[s].mean():.0%} => {'ADD' if q3 else 'no'}")
aps = A7
if q3:
    p8 = apply(fit(train, A7 + ["LIRF"], params), au)
    ok = rm(au.y - p8) < rm(au.y - p_q2)
    out["Q3"]["combined"] = (rm(au.y - p8), rm(au.y - p_q2), bool(ok))
    print(f"   combined 8 airports {rm(au.y - p8):.2f} vs 7 airports {rm(au.y - p_q2):.2f} => {'PASS' if ok else 'FAIL'}")
    if ok:
        aps = A7 + ["LIRF"]

out["v46"] = {"airports": aps, "train": "Jan+Jul+Sep+Oct+Nov+Dec" if q1 else "Jan+Jul+Nov+Dec", "params": params,
              "differs_from_v45": bool(q1 or q2 or aps != A7)}
print(f"\nv46 recipe: {aps}, training {out['v46']['train']}, {'capacity variant' if q2 else 'v45 params'} -> "
      f"{'BUILD v46' if out['v46']['differs_from_v45'] else 'same as v45, nothing to build'}")
(HERE / "evaluate_v46.json").write_text(json.dumps(out, indent=2, default=float))
