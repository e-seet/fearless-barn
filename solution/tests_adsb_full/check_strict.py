#!/usr/bin/env python3
"""Exploratory (not the test): strict-matched ADS-B estimates vs the label and vs the model's
residual (seasonal_full MLP, trained without Jan/Jul), on the fetched 2025 days; plus like-for-like
coverage on any 2026 day given. usage: check_strict.py 2025-01-15 2025-07-15 2026-01-15"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import fetch, features
from common import dataset as D
from tests_tail_diagnosis.diagnose import CACHE

X, y, meta = D.load_part("seasonal_full", "test", "with_aobt", filter_target=False)
pred = pd.Series(np.load(CACHE / "mlp6_seasonal_full_test.npy"), index=X.index)
y = pd.Series(np.asarray(y, float), index=X.index)
EST = ("e_first", "e_disp", "e_move")

for day in sys.argv[1:]:
    t = fetch.targets(day)
    t = t[t["T"] < pd.Timestamp(day).value / 1e9 + 86400].set_index("MVT_ID_mvt")
    e = features.load_day(day).reindex(t.index)
    cov = e.e_first.notna().groupby(t.ADEP_mvt).mean()
    print(f"\n== {day}: {len(t)} deps | strict-matched ground coverage {e.e_first.notna().mean():.3f}")
    print("   coverage by airport:", {k: round(v, 2) for k, v in cov.items()})
    if day.startswith("2026"):
        continue
    ids = e.index[e.e_first.notna()].intersection(X.index)
    ids = ids[np.isfinite(y[ids])]
    df = e.loc[ids].copy(); df["y"] = y[ids]; df["p"] = pred[ids]; df["r"] = df.y - df.p
    df["ap"] = t.ADEP_mvt.reindex(ids).values
    for a, s in [("ALL", df)] + list(df.groupby("ap")):
        if len(s) < 30:
            continue
        line = f"   {a:5s} n={len(s):4d} model rmse {np.sqrt((s.r ** 2).mean()):6.1f} sd_y {s.y.std():6.1f} |"
        for c in EST:
            v = s[c].notna()
            d = s[c][v] - s.y[v]
            z = s[c][v] - s.p[v]
            k = (z * s.r[v]).sum() / (z * z).sum()
            line += (f" {c}: err med {d.median():+5.0f} rmse {np.sqrt((d ** 2).mean()):5.0f} "
                     f"corr(y) {np.corrcoef(s[c][v], s.y[v])[0, 1]:.2f} corr(resid) {np.corrcoef(z, s.r[v])[0, 1]:+.2f} "
                     f"fix {np.sqrt((s.r[v] ** 2).mean()):4.0f}->{np.sqrt(((s.r[v] - k * z) ** 2).mean()):4.0f} |")
        print(line)
