#!/usr/bin/env python3
"""PREREGISTERED_v45.md: Sep-Oct 2025 fresh surface. Q1 bigger training set, Q2 LSZH/LEMD/EGLL."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((HERE / "model_v44" / "config.json").read_text())
A4, FEAT, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
CAND = ["LSZH", "LEMD", "EGLL"]
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
_est = {}


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


def fit(tr, aps, shuffle=None):
    tr = cat(tr[tr.has & tr.ap.isin(aps)], aps)
    if shuffle is not None:
        rng = np.random.default_rng(shuffle)
        for a in aps:
            idx = np.flatnonzero(tr.ap.to_numpy() == a)
            tr.iloc[idx, [tr.columns.get_loc(c) for c in FEAT_ADSB]] = tr.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    return lgb.LGBMRegressor(**PARAMS).fit(tr[FEAT], tr.y - tr.base), aps


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


# training surfaces (held-out bases)
ftt_sf = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
jj, _ = frame("seasonal_full", "test", (np.load(CACHE / "tree_seasonal_full_test.npy") + np.load(CACHE / "mlp6_seasonal_full_test.npy") + ftt_sf) / 3,
              {"2025-01", "2025-07"})
z = np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")
nd, _ = frame("forward", "test", (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy") + z["pt"]) / 3,
              {"2025-11", "2025-12"})
so, so_days = frame("forward", "val", (np.load(CACHE / "tree_forward_val.npy") + np.load(CACHE / "mlp12_forward_val.npy") + z["pv"]) / 3,
                    {"2025-09", "2025-10"})
big = pd.concat([jj, nd])
print(f"Sep-Oct: {len(so_days)} of 61 days, {len(so)} rows; covered by airport:", so.groupby("ap").has.mean().round(2).to_dict())
out = {"sepoct_days": len(so_days)}
b0 = rm(so.y - so.base)


def perday(p, mask=None):
    m = np.ones(len(so), bool) if mask is None else mask
    return {d: rm((so.y - p)[m & (so.day == d).to_numpy()]) - rm((so.y - so.base)[m & (so.day == d).to_numpy()]) for d in sorted(so.day.unique())}


# ---------------- Q1: R44 (saved) vs R45 (Jan+Jul+Nov+Dec) ----------------
booster = lgb.Booster(model_file=str(HERE / "model_v44" / "adsb_corrector.txt"))
p44 = apply((None, A4), so, predict=booster.predict)
r45 = fit(big, A4); p45 = apply(r45, so)
d44, d45 = perday(p44), perday(p45)
wins = sum(d45[d] < d44[d] for d in d44)
q1 = rm(so.y - p45) < rm(so.y - p44) and wins >= len(d44) / 2
out["Q1"] = {"base": b0, "R44": rm(so.y - p44), "R45": rm(so.y - p45), "R45_day_wins": wins, "n_days": len(d44),
             "R44_days_better_than_base": sum(v < 0 for v in d44.values()), "ADOPT_R45": bool(q1),
             "per_airport": {a: (rm((so.y - so.base)[so.ap == a]), rm((so.y - p44)[so.ap == a]), rm((so.y - p45)[so.ap == a])) for a in A4}}
print(f"\nQ1: base {b0:.2f} | R44 (saved v44 corrector) {rm(so.y - p44):.2f} ({rm(so.y - p44) - b0:+.2f}, "
      f"{out['Q1']['R44_days_better_than_base']}/{len(d44)} days better than base) | R45 {rm(so.y - p45):.2f} "
      f"(wins {wins}/{len(d44)} days vs R44) => {'ADOPT R45' if q1 else 'keep R44'}")
print("   per airport base/R44/R45:", {a: tuple(round(x, 1) for x in v) for a, v in out["Q1"]["per_airport"].items()})
train = big if q1 else jj

# ---------------- Q2: LSZH, LEMD, EGLL ----------------
out["Q2"] = {}; added = []
for X_ in CAND:
    s = (so.ap == X_).to_numpy()
    m = fit(big, [X_]); p = apply(m, so)
    br, nr = rm((so.y - so.base)[s]), rm((so.y - p)[s])
    pd_ = perday(p, s)
    perms = [rm((so.y - apply(fit(big, [X_], shuffle=k), shuffled(so, [X_], k)))[s]) for k in range(5)]
    ok = nr < br and sum(v < 0 for v in pd_.values()) >= 2 / 3 * len(pd_) and nr < min(perms)
    out["Q2"][X_] = {"base": br, "new": nr, "days_better": sum(v < 0 for v in pd_.values()), "perm_best": min(perms),
                     "covered": float(so.has[s].mean()), "n_train_covered": int((big.has & (big.ap == X_)).sum()), "PASS": bool(ok)}
    if ok:
        added.append(X_)
    print(f"Q2 {X_}: Sep-Oct {br:.1f} -> {nr:.1f}, days better {out['Q2'][X_]['days_better']}/{len(pd_)}, permuted best {min(perms):.1f}, "
          f"covered {so.has[s].mean():.0%}, training rows {out['Q2'][X_]['n_train_covered']} => {'ADD' if ok else 'no'}")

# ---------------- combined ----------------
chosen = rm(so.y - (p45 if q1 else p44))
if added:
    U = A4 + added
    pU = apply(fit(train, U), so)
    out["combined"] = {"airports": U, "train": "Jan+Jul+Nov+Dec" if q1 else "Jan+Jul", "sepoct": rm(so.y - pU), "four_airport": chosen,
                       "PASS": bool(rm(so.y - pU) < chosen)}
    print(f"\ncombined {U} ({out['combined']['train']}): Sep-Oct {rm(so.y - pU):.2f} vs 4-airport {chosen:.2f} => "
          f"{'PASS' if out['combined']['PASS'] else 'FAIL'}")
final_aps = A4 + added if (added and out["combined"]["PASS"]) else A4
out["v45"] = {"airports": final_aps, "train": "Jan+Jul+Nov+Dec" if q1 else "Jan+Jul",
              "differs_from_v44": bool(q1 or final_aps != A4)}
print(f"\nv45 recipe: airports {final_aps}, training {out['v45']['train']} -> "
      f"{'BUILD v45' if out['v45']['differs_from_v44'] else 'same as v44, nothing to build'}")
(HERE / "evaluate_v45.json").write_text(json.dumps(out, indent=2, default=float))
