#!/usr/bin/env python3
"""Exploratory: does the ADS-B first-motion time match the true off-block on 2025 days?
Estimate = seconds from the first ground point moving >= 3 kt to takeoff. Not a model test."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import fetch
from common import io as cio

r = cio.load_training(columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"])
y = pd.to_numeric(r.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt, errors="coerce")
for day in sys.argv[1:]:
    p = pd.read_parquet(HERE / "points" / f"{day}.parquet"); t = fetch.targets(day)
    t = t[t["T"] < pd.Timestamp(day).value / 1e9 + 86400].set_index("MVT_ID_mvt"); t["y"] = y
    g = p[p.ground & (p.t_rel < 0)]
    t["e_move"] = -g[g.gs_kt >= 3].groupby("MVT_ID_mvt").t_rel.min()
    c = t[t.e_move.notna() & t.y.notna()]
    print(f"\n== {day}: {len(t)} deps, ground-moving coverage {t.e_move.notna().mean():.3f}")
    for ap, s in [("ALL", c)] + list(c.groupby("ADEP_mvt")):
        if len(s) < 20:
            continue
        d = s.e_move - s.y
        cov = t[t.ADEP_mvt == ap].e_move.notna().mean() if ap != "ALL" else t.e_move.notna().mean()
        print(f"{ap:5s} n={len(s):4d} cov={cov:.2f} corr={np.corrcoef(s.e_move, s.y)[0, 1]:.3f} "
              f"med|err|={d.abs().median():6.1f} within60s={(d.abs() <= 60).mean():.2f} bias={d.median():+6.1f} "
              f"rmse_est={np.sqrt((d ** 2).mean()):7.1f} sd_y={s.y.std():7.1f} y_mean={s.y.mean():6.0f}")
