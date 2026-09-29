#!/usr/bin/env python3
"""ADS-B corrector, cross-month Jan <-> Jul, with a permutation control (PREREGISTERED.md)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import dataset as D
from tests_tail_diagnosis.diagnose import CACHE

JAN = ["2025-01-%02d" % d for d in (6, 9, 13, 15, 18, 21, 24, 28, 31)]
JUL = ["2025-07-%02d" % d for d in (3, 7, 10, 15, 17, 20, 23, 26, 30)]
ADSB = ["e_first", "e_disp", "e_move", "n_ground", "liftoff_dt"]
PARAMS = dict(objective="l2", learning_rate=0.03, n_estimators=300, num_leaves=15, min_data_in_leaf=50,
              feature_fraction=0.8, random_state=42, verbose=-1)
rm = lambda r: float(np.sqrt(np.mean(np.square(r))))

X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
m = np.load(ROOT / "tests_ftt" / "meta_seasonal_full.npz", allow_pickle=True)
assert (np.asarray(m["test_ids"]).astype(str) == X.index.astype(str)).all(), "row order mismatch"
ap = np.asarray(meta["ADEP_mvt"]).astype(str)
tree = np.load(CACHE / "tree_seasonal_full_test.npy"); mlp = np.load(CACHE / "mlp6_seasonal_full_test.npy")
ftt = np.mean([np.load(ROOT / "tests_ftt" / f"preds_seed{s}_seasonal_full_lr1e-3.npz")["pt"] for s in range(42, 51)], axis=0)
tab = np.mean([np.load(ROOT / "tests_tabnet" / f"preds_seed{s}_sf_lr2e-3p.npz")["pt"] for s in (42, 43, 44)], axis=0)
base = (tree + mlp + ftt) / 3
base[ap == "LIRF"] = 0.541 * mlp[ap == "LIRF"] + 0.459 * tab[ap == "LIRF"]
df = pd.DataFrame({"y": np.asarray(y, float), "base": base, "ap": ap}, index=X.index)
from common import io as cio
raw = cio.load_training(columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).set_index("MVT_ID_mvt")
t = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True)
df["day"] = np.asarray(t.dt.strftime("%Y-%m-%d")); df["hour"] = np.asarray(t.dt.hour)
print(f"base (held out) Jan+Jul RMSE {rm(df.y[np.isfinite(df.y)] - df.base[np.isfinite(df.y)]):.2f}")

# ADS-B estimates; a takeoff can draw points from its own day and the previous day's file
est = []
for d in JAN + JUL:
    f = HERE / "points" / f"{d}.parquet"
    if f.exists():
        est.append(features.load_day(d))
    else:
        print(f"missing {d}")
e = pd.concat(est).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
days = {d for d in JAN + JUL if (HERE / "points" / f"{d}.parquet").exists()}
print(f"{len(days)} of {len(JAN + JUL)} days fetched")
df = df[df.day.isin(days) & np.isfinite(df.y)].join(e[ADSB])
df["covered"] = df.e_first.notna()
df["apc"] = df.ap.astype("category")
df["r"] = df.y - df.base
df["mon"] = df.day.str[5:7]
print(df.groupby("mon").covered.agg(["size", "mean"]))

FEAT = ADSB + ["base", "apc", "hour"]


def run(train, test, shuffle_seed=None):
    tr = train[train.covered].copy(); te = test.copy()
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        for part in (tr, te):
            cov = part.covered.to_numpy()
            for a in part.ap.unique():
                idx = np.flatnonzero(cov & (part.ap.to_numpy() == a))
                perm = rng.permutation(idx)
                part.iloc[idx, [part.columns.get_loc(c) for c in ADSB]] = part.iloc[perm][ADSB].to_numpy()
    mdl = lgb.LGBMRegressor(**PARAMS).fit(tr[FEAT], tr.r)
    pred = te.base.to_numpy().copy()
    c = te.covered.to_numpy()
    pred[c] += mdl.predict(te.loc[c, FEAT])
    return rm(te.y - te.base), rm(te.y - pred), pred


out = {}
for name, a, b in (("jan_to_jul", "01", "07"), ("jul_to_jan", "07", "01")):
    tr, te = df[df.mon == a], df[df.mon == b]
    b0, b1, pred = run(tr, te)
    perms = [run(tr, te, s)[1] for s in range(5)]
    per_ap = {x: (rm(te.y[te.ap == x] - te.base[te.ap == x]), rm((te.y - pred)[te.ap == x])) for x in sorted(te.ap.unique())}
    out[name] = {"n": len(te), "covered": float(te.covered.mean()), "base": b0, "adsb": b1, "gain": b1 - b0,
                 "perm": perms, "perm_best_gain": min(perms) - b0, "per_airport": per_ap}
    print(f"\n== {name}: {len(te)} rows ({te.covered.mean():.0%} covered) | pooled {b0:.2f} -> {b1:.2f} ({b1-b0:+.2f}) "
          f"| permuted best {min(perms):.2f} ({min(perms)-b0:+.2f}), mean {np.mean(perms):.2f}")
    for x, (u, v) in per_ap.items():
        print(f"   {x}: {u:6.1f} -> {v:6.1f} ({v-u:+.1f})")

g = [out[k]["gain"] for k in ("jan_to_jul", "jul_to_jan")]
c1 = all(v < 0 for v in g)
c2 = all(out[k]["adsb"] < min(out[k]["perm"]) for k in ("jan_to_jul", "jul_to_jan"))
c3 = np.mean(g) <= -0.3
out["verdict"] = {"both_directions_improve": c1, "beats_best_permutation": c2, "mean_gain_ge_0.3": bool(c3), "PASS": bool(c1 and c2 and c3)}
print(f"\nPRE-REGISTERED: both directions {c1} | beats permutation {c2} | mean gain {np.mean(g):+.2f} (<= -0.3: {c3}) "
      f"=> {'PASS' if (c1 and c2 and c3) else 'FAIL'}")
(HERE / "evaluate.json").write_text(json.dumps(out, indent=2, default=float))
