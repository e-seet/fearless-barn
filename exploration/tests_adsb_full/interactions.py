#!/usr/bin/env python3
"""PREREGISTERED_interactions.md: leave-one-month-out, A (v46 inputs) vs B (+ all tabular columns) vs B_perm."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from common import io as cio
from common import features as F
from tests_tail_diagnosis.diagnose import CACHE

cfg = json.loads((HERE / "model_v46" / "config.json").read_text())
A7, FEAT_A, FEAT_ADSB, PARAMS = cfg["airports"], cfg["features"], cfg["adsb_features"], cfg["params"]
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
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
    assert len(base) == len(X)
    t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
    df = X.copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str)
    df["y"] = np.asarray(y, float); df["base"] = base; df["ap"] = np.asarray(meta["ADEP_mvt"]).astype(str)
    df["day"] = np.asarray(t.dt.strftime("%Y-%m-%d")); df["hour"] = np.asarray(t.dt.hour)
    days = sorted(f.stem for f in (HERE / "points").glob("*.parquet") if f.stem[:7] in prefixes)
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[FEAT_ADSB])
    frames.append(df)
df = pd.concat(frames)
df = df[~df.index.duplicated()]
df["mon"] = df.day.str[5:7]
df["has"] = df.e_first.notna() & df.ap.isin(A7)
TAB = [c for c in frames[0].columns if c not in ("y", "base", "ap", "day", "hour") and c not in FEAT_ADSB]
for c in TAB:
    if df[c].dtype == object:
        df[c] = df[c].astype("category")
df["apc"] = pd.Categorical(df.ap.where(df.ap.isin(A7)), categories=A7)
FEAT_B = FEAT_A + [c for c in TAB if c not in FEAT_A]
print(f"{len(df)} rows over months {sorted(df.mon.unique())}; covered at 7 airports {int(df.has.sum())}; "
      f"B adds {len(FEAT_B) - len(FEAT_A)} tabular columns")


def shuffle_adsb(part, seed):
    part = part.copy(); rng = np.random.default_rng(seed); has = part.has.to_numpy()
    for a in A7:
        idx = np.flatnonzero(has & (part.ap.to_numpy() == a))
        part.iloc[idx, [part.columns.get_loc(c) for c in FEAT_ADSB]] = part.iloc[rng.permutation(idx)][FEAT_ADSB].to_numpy()
    return part


def run(tr, te, feat):
    trc = tr[tr.has]
    m = lgb.LGBMRegressor(**PARAMS).fit(trc[feat], trc.y - trc.base)
    p = te.base.to_numpy().copy(); c = te.has.to_numpy()
    p[c] += m.predict(te.loc[c, feat])
    return p


res = {}; se = {"A": [], "B": [], "B_perm": [], "base": []}
for mon in sorted(df.mon.unique()):
    tr, te = df[df.mon != mon], df[df.mon == mon]
    pA, pB = run(tr, te, FEAT_A), run(tr, te, FEAT_B)
    pP = run(shuffle_adsb(tr, 1), shuffle_adsb(te, 2), FEAT_B)
    res[mon] = {"n": len(te), "base": rm(te.y - te.base), "A": rm(te.y - pA), "B": rm(te.y - pB), "B_perm": rm(te.y - pP)}
    for k, p in (("A", pA), ("B", pB), ("B_perm", pP), ("base", te.base.to_numpy())):
        se[k].append(((te.y - p) ** 2).to_numpy())
    r = res[mon]
    print(f"  month {mon}: base {r['base']:.2f} | A {r['A']:.2f} | B {r['B']:.2f} ({r['B'] - r['A']:+.2f} vs A) | B_perm {r['B_perm']:.2f}", flush=True)
pooled = {k: float(np.sqrt(np.concatenate(v).mean())) for k, v in se.items()}
wins = sum(r["B"] < r["A"] for r in res.values())
c1, c2, c3 = pooled["B"] < pooled["A"], wins >= 5, pooled["B"] < pooled["B_perm"]
print(f"\nPOOLED: base {pooled['base']:.2f} | A {pooled['A']:.2f} | B {pooled['B']:.2f} ({pooled['B'] - pooled['A']:+.2f}) | B_perm {pooled['B_perm']:.2f}")
print(f"PRE-REGISTERED: B<A pooled {c1} | B<A in >=5/7 months {c2} ({wins}/7) | B<B_perm {c3} => {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
(HERE / "interactions.json").write_text(json.dumps({"months": res, "pooled": pooled, "wins": wins,
                                                    "PASS": bool(c1 and c2 and c3)}, indent=2, default=float))
