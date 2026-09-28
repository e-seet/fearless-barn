#!/usr/bin/env python3
"""Tail study, step 1: what do the tail rows look like across all of 2025? Read-only."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import config
from common import dataset as D
pd.set_option("display.width", 220, "display.max_columns", 30)
X = D.build_scope("train2025", "with_aobt").set_index(config.ID)
y = D._targets().reindex(X.index)
df = pd.DataFrame({"y": y.to_numpy(), "ap": X["ADEP_mvt"].astype(str).to_numpy() if "ADEP_mvt" in X else None,
                   "month": X["month_num"].to_numpy(), "rec": np.asarray(X["has_flight_record"]) > 0,
                   "ds": np.asarray(X["delta_sched_sec"], float), "da": np.asarray(X["delta_aobt_sec"], float),
                   "has_aobt": np.asarray(X["has_aobt"]) > 0}, index=X.index)
df = df[np.isfinite(df.y) & (df.y > 0)]
print("rows", len(df))
bands = [(0, 7200, "inband"), (7200, 25200, "7k-25k"), (25200, 80000, "25k-80k"), (80000, 1e9, ">=80k")]
df["band"] = pd.cut(df.y, [b[0] for b in bands] + [1e9], labels=[b[2] for b in bands], right=True)
print("\n== band x airport (counts) ==")
print(pd.crosstab(df.band, df.ap))
print("\n== band x flight record ==")
print(pd.crosstab(df.band, df.rec))
art = df[df.y >= 80000]
print("\n== all >=80k rows ==")
print(art.assign(y_minus_day=art.y - 86400, ds_minus_y=art.ds - art.y).sort_values(["ap", "month"]).to_string())
print("\n== LIRF, no record: y vs ds, by ds bin ==")
L = df[(df.ap == "LIRF") & (~df.rec)].copy()
L["dsbin"] = pd.cut(L.ds, [-1e9, 3600, 7200, 25200, 40000, 50000, 60000, 70000, 80000, 90000, 1e9])
def summ(g):
    return pd.Series({"n": len(g), "identity_60s": float((abs(g.y - g.ds) <= 60).mean()),
                      "artifact_>=80k": float((g.y >= 80000).mean()), "y_med": g.y.median(),
                      "y_mean": g.y.mean(), "ds_med": g.ds.median()})
print(L.groupby("dsbin", observed=True).apply(summ).to_string())
print("\n== non-LIRF, no record: same ==")
N = df[(df.ap != "LIRF") & (~df.rec)].copy()
N["dsbin"] = pd.cut(N.ds, [-1e9, 3600, 7200, 25200, 50000, 70000, 1e9])
print(N.groupby("dsbin", observed=True).apply(summ).to_string())
