#!/usr/bin/env python3
"""v63 step 1: member M on the ranking rows (classifier + normal regressor trained on all 2025, seed 42 as tested),
per-airport weights fitted on all 7 held-out months, and the 2026 drift diagnostics (advisor item 3)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from common import dataset as D
from common import io as cio
from copymix import CLS, REG, times, mixture, EPOCH

raw = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt", "BLOCK_TIME_UTC_mvt"])
raw = raw[raw.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
X, y, _ = D.load_part("full", "train", "with_aobt", filter_target=False)
y = pd.Series(np.asarray(y, float), index=X.index); ok = np.isfinite(y.to_numpy())
blk = (pd.to_datetime(raw.BLOCK_TIME_UTC_mvt.reindex(X.index), utc=True) - pd.to_datetime(raw.SCHED_TIME_UTC_mvt.reindex(X.index), utc=True)).dt.total_seconds()
copy = (blk.abs() <= 60).to_numpy()
cls = lgb.LGBMClassifier(**CLS).fit(X[ok], copy[ok])
reg = lgb.LGBMRegressor(**REG).fit(X[ok & ~copy], np.clip(y[ok & ~copy], 0, 7200))

Xr, meta = D.load_ranking_features("with_aobt")
Xr = D.align_columns(X, Xr)
rk = cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "LOBT_flt"])
rk = rk[rk.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
mv, sc, lb = times(Xr.index, rk)
p = cls.predict_proba(Xr)[:, 1]; yn = reg.predict(Xr)
M = mixture(p, yn, mv, sc, lb)
ap = rk.ADEP_mvt.reindex(Xr.index).astype(str).to_numpy()
mon = pd.to_datetime(rk.MVT_TIME_UTC_mvt.reindex(Xr.index), utc=True).dt.month.to_numpy()
out = pd.DataFrame({"M": M, "p": p, "yn": yn, "ap": ap, "mon": mon, "mvt": mv, "sched": sc, "lobt": lb}, index=Xr.index)
out.to_parquet(HERE / "cache" / "M_ranking.parquet")
print(f"ranking rows {len(out)}; M NaN {int(np.isnan(M).sum())}")

# weights on all 7 held-out months
d = pd.read_parquet(HERE / "cache" / "F_heldout.parquet").join(pd.read_parquet(HERE / "cache" / "M_heldout.parquet"))
d["y"] = y.reindex(d.index).to_numpy(); d = d[np.isfinite(d.y)]
W = {}
for a, g in d.groupby("ap"):
    dl = (g.M - g.F).to_numpy(); r = (g.y - g.F).to_numpy()
    W[a] = float(np.clip(dl @ r / (dl @ dl), 0, 1)) if dl @ dl > 0 else 0.0
print("weights:", {a: round(v, 3) for a, v in W.items()})

# drift: per airport, mean p on ranking Jan/Jul vs the 2025 copy rate in Jan/Jul; MVT-SCHED quantiles
ap25 = raw.ADEP_mvt.reindex(X.index).astype(str).to_numpy()
mon25 = pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True).dt.month.to_numpy()
ms25 = (pd.to_datetime(raw.MVT_TIME_UTC_mvt.reindex(X.index), utc=True) - pd.to_datetime(raw.SCHED_TIME_UTC_mvt.reindex(X.index), utc=True)).dt.total_seconds().to_numpy()
rows = []
for a in sorted(W):
    for m in (1, 7):
        k25 = (ap25 == a) & (mon25 == m) & ok; kr = (ap == a) & (mon == m)
        rows.append({"ap": a, "mon": m, "w": round(W[a], 3), "copy25": round(copy[k25].mean(), 4), "p2025_fit": round(cls.predict_proba(X[k25])[:, 1].mean(), 4),
                     "p_rank": round(p[kr].mean(), 4), "ms25_med": float(np.nanmedian(ms25[k25])), "ms_rank_med": float(np.nanmedian((mv - sc)[kr])),
                     "ms25_p90": float(np.nanpercentile(ms25[k25], 90)), "ms_rank_p90": float(np.nanpercentile((mv - sc)[kr], 90))})
dr = pd.DataFrame(rows); print(dr.to_string(index=False))
(HERE / "weights_v63.json").write_text(json.dumps({"weights_per_airport": W, "drift": rows}, indent=2, default=float))
print("ALL DONE")
