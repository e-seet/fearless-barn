#!/usr/bin/env python3
"""PREREGISTERED_extension.md: (A) frozen v43 corrector on Nov-Dec 2025; (B) per-airport extension."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from common import io as cio
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((HERE / "model" / "config.json").read_text())
SHIP, FEAT, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
CAND = ["EDDM", "LSZH", "EGLL", "LEMD"]
ALLAP = SHIP + CAND
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")


def frame(split, part, base):
    X, y, meta = D.load_part(split, part, "with_aobt", filter_target=False)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    return pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": np.asarray(meta["ADEP_mvt"]).astype(str),
                         "day": np.asarray(t.dt.strftime("%Y-%m-%d")), "hour": np.asarray(t.dt.hour)}, index=X.index)


def attach(df, prefixes):
    days = sorted(f.stem for f in (HERE / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB])
    df["has"] = df.e_first.notna()
    return df, days


def cat(df, aps):
    df = df.copy(); df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(aps)), categories=aps); return df


# Jan + Jul 2025 training rows (held-out seasonal_full base, as for v43)
tree = np.load(CACHE / "tree_seasonal_full_test.npy"); mlp = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
jj, _ = attach(frame("seasonal_full", "test", (tree + mlp + ftt) / 3), {"2025-01", "2025-07"})
jj["mon"] = jj.day.str[5:7]
# Nov-Dec 2025 (forward test; models trained Jan-Aug). Only FTT seed 42 exists on the forward split.
fb = (np.load(CACHE / "tree_forward_test.npy") + np.load(CACHE / "mlp12_forward_test.npy")
      + np.load(ROOT / "tests_ftt" / "preds_seed42_lr1e-3.npz")["pt"]) / 3
nd, nd_days = attach(frame("forward", "test", fb), {"2025-11", "2025-12"})
print(f"Nov-Dec: {len(nd_days)} of 61 days fetched, {len(nd)} rows; covered by airport:",
      nd.groupby("ap").has.mean().round(2).to_dict())
out = {"novdec_days": len(nd_days)}


def apply(model, te, aps):
    te = cat(te, model_aps[id(model)] if id(model) in model_aps else aps)
    c = (te.has & te.ap.isin(aps)).to_numpy()
    p = te.base.to_numpy().copy()
    if c.any():
        p[c] += model.predict(te.loc[c, FEAT])
    return p, c
model_aps = {}


def fit(tr, aps, shuffle=None):
    tr = cat(tr[tr.has & tr.ap.isin(aps)], aps)
    if shuffle is not None:
        rng = np.random.default_rng(shuffle)
        for a in aps:
            idx = np.flatnonzero(tr.ap.to_numpy() == a)
            tr.iloc[idx, [tr.columns.get_loc(c) for c in FEAT_ADSB]] = tr.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    m = lgb.LGBMRegressor(**PARAMS).fit(tr[FEAT], tr.y - tr.base)
    model_aps[id(m)] = aps
    return m


def shuffled(te, aps, seed):
    te = te.copy(); rng = np.random.default_rng(seed); has = te.has.to_numpy()
    for a in aps:
        idx = np.flatnonzero(has & (te.ap.to_numpy() == a))
        te.iloc[idx, [te.columns.get_loc(c) for c in FEAT_ADSB]] = te.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    return te


# ---------------- A: frozen v43 corrector on Nov-Dec ----------------
booster = lgb.Booster(model_file=str(HERE / "model" / "adsb_corrector.txt"))
class Frozen:
    def predict(self, X): return booster.predict(X)
fz = Frozen(); model_aps[id(fz)] = SHIP
pA, cA = apply(fz, nd, SHIP)
se0, se1 = (nd.y - nd.base) ** 2, (nd.y - pA) ** 2
g = (se0 - se1).to_numpy(); j = np.argmax(g); keep = np.ones(len(nd), bool); keep[j] = False
perday = {d: rm((nd.y - pA)[nd.day == d]) - rm((nd.y - nd.base)[nd.day == d]) for d in sorted(nd.day.unique())}
A = {"pooled": (rm(nd.y - nd.base), rm(nd.y - pA)), "drop_best": (float(np.sqrt(se0[keep].mean())), float(np.sqrt(se1[keep].mean()))),
     "days_better": sum(v < 0 for v in perday.values()), "n_days": len(perday), "n_corrected": int(cA.sum()),
     "per_airport": {a: (rm((nd.y - nd.base)[nd.ap == a]), rm((nd.y - pA)[nd.ap == a])) for a in SHIP}}
A["PASS"] = bool(A["pooled"][1] < A["pooled"][0] and A["days_better"] >= 2 / 3 * A["n_days"] and A["drop_best"][1] < A["drop_best"][0])
out["A"] = A
print(f"\nA (frozen v43 corrector on Nov-Dec): pooled {A['pooled'][0]:.2f} -> {A['pooled'][1]:.2f} ({A['pooled'][1]-A['pooled'][0]:+.2f}) | "
      f"best row removed {A['drop_best'][1]-A['drop_best'][0]:+.2f} | days better {A['days_better']}/{A['n_days']} | {cA.sum()} rows corrected")
print("   per airport:", {a: f"{u:.1f}->{v:.1f}" for a, (u, v) in A["per_airport"].items()}, "=>", "PASS" if A["PASS"] else "FAIL")

# ---------------- B: per-airport extension ----------------
out["B"] = {}; added = []
for X_ in CAND:
    r = {}
    ok1 = True
    for a, b in (("01", "07"), ("07", "01")):
        m = fit(jj[jj.mon == a], [X_]); te = jj[(jj.mon == b)]
        p, _ = apply(m, te, [X_]); s = te.ap == X_
        r[f"{a}->{b}"] = (rm((te.y - te.base)[s]), rm((te.y - p)[s]), int((te.has & s).sum()))
        ok1 &= r[f"{a}->{b}"][1] < r[f"{a}->{b}"][0]
    m = fit(jj, [X_]); p, _ = apply(m, nd, [X_]); s = (nd.ap == X_).to_numpy()
    base_r, new_r = rm((nd.y - nd.base)[s]), rm((nd.y - p)[s])
    pd_ = [rm((nd.y - p)[s & (nd.day == d).to_numpy()]) - rm((nd.y - nd.base)[s & (nd.day == d).to_numpy()]) for d in sorted(nd.day.unique())]
    perms = [rm((nd.y - apply(fit(jj, [X_], shuffle=k), shuffled(nd, [X_], k), [X_])[0])[s]) for k in range(5)]
    ok2 = new_r < base_r and sum(v < 0 for v in pd_) >= 2 / 3 * len(pd_)
    ok3 = new_r < min(perms)
    r.update({"novdec": (base_r, new_r), "novdec_days_better": sum(v < 0 for v in pd_), "perm_best": min(perms),
              "novdec_covered": float(nd.has[s].mean()), "PASS": bool(ok1 and ok2 and ok3)})
    out["B"][X_] = r
    if r["PASS"]:
        added.append(X_)
    print(f"\nB {X_}: Jan->Jul {r['01->07'][0]:.1f}->{r['01->07'][1]:.1f} (n={r['01->07'][2]}), Jul->Jan {r['07->01'][0]:.1f}->{r['07->01'][1]:.1f} (n={r['07->01'][2]}) | "
          f"Nov-Dec {base_r:.1f}->{new_r:.1f}, days better {r['novdec_days_better']}/{len(pd_)}, permuted best {min(perms):.1f}, covered {r['novdec_covered']:.0%} => {'ADD' if r['PASS'] else 'no'}")

# ---------------- v44 candidate check: union refit vs frozen v43 corrector on Nov-Dec ----------------
if added:
    U = SHIP + added
    mU = fit(jj, U); pU, cU = apply(mU, nd, U)
    out["v44"] = {"airports": U, "novdec_pooled": rm(nd.y - pU), "v43_novdec_pooled": A["pooled"][1], "n_corrected": int(cU.sum()),
                  "PASS": bool(rm(nd.y - pU) <= A["pooled"][1])}
    print(f"\nv44 check: union {U} refit -> Nov-Dec pooled {rm(nd.y - pU):.2f} vs frozen v43 {A['pooled'][1]:.2f} "
          f"=> {'PASS (build v44)' if out['v44']['PASS'] else 'FAIL (keep v43)'}")
else:
    print("\nno airport added -> v43 stays the ADS-B candidate")
(HERE / "evaluate_ext.json").write_text(json.dumps(out, indent=2, default=float))
