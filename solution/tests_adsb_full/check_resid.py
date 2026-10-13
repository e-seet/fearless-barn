#!/usr/bin/env python3
"""Exploratory: does the ADS-B first-motion estimate explain the model's error (seasonal_full MLP,
trained without Jan/Jul) on the fetched 2025 days? Corr of residual y-pred with (estimate - pred)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import fetch
from common import dataset as D
from tests_tail_diagnosis.diagnose import CACHE
X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
pred = pd.Series(np.load(CACHE / "mlp6_seasonal_full_test.npy"), index=X.index)
y = pd.Series(np.asarray(y, float), index=X.index)
for day in sys.argv[1:]:
    p = pd.read_parquet(HERE / "points" / f"{day}.parquet")
    g = p[p.ground & (p.t_rel < 0)]
    e = -g[g.gs_kt >= 3].groupby("MVT_ID_mvt").t_rel.min()
    ids = e.index.intersection(X.index)
    ids = ids[np.isfinite(y[ids])]
    ap = meta["ADEP_mvt"].astype(str).reindex(ids) if hasattr(meta["ADEP_mvt"], "reindex") else pd.Series(meta["ADEP_mvt"], index=X.index)[ids]
    df = pd.DataFrame({"y": y[ids], "p": pred[ids], "e": e[ids], "ap": ap.values}, index=ids)
    df["r"] = df.y - df.p
    print(f"\n== {day}: {len(df)} covered rows with labels")
    for a, s in [("ALL", df)] + list(df.groupby("ap")):
        if len(s) < 30: continue
        # clipped estimate to limit wrong-match outliers; best single-slope correction on this day (optimistic)
        z = np.clip(s.e, 0, 3600) - s.p
        c = np.corrcoef(z, s.r)[0, 1]
        k = (z * s.r).sum() / (z * z).sum()
        new = s.r - k * z
        print(f"{a:5s} n={len(s):4d} corr(resid, est-pred)={c:+.3f} rmse model {np.sqrt((s.r**2).mean()):6.1f} -> in-sample slope fix {np.sqrt((new**2).mean()):6.1f} (k={k:+.2f})")
