#!/usr/bin/env python3
"""Carrier-echo study, step 2: within LIRF no-record departures, does the carrier separate echo
from non-echo, stably across months? Leave-one-month-out carrier rates (honest)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common import io as cio
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 80)
r = cio.load_training(); d = r[r.PHASE_mvt.eq("DEP") & r.ADEP_mvt.eq("LIRF") & r.FLIGHT_ID_mvt.isna()].copy()
t = lambda c: pd.to_datetime(d[c], utc=True, errors="coerce")
d["y"] = pd.to_numeric(d.TAXITIME_SEC_mvt, errors="coerce")
d["ds"] = (t("MVT_TIME_UTC_mvt") - t("SCHED_TIME_UTC_mvt")).dt.total_seconds()
d["echo"] = ((t("BLOCK_TIME_UTC_mvt") - t("SCHED_TIME_UTC_mvt")).dt.total_seconds().abs() <= 60)
d["carrier"] = d.FLIGHT_mvt.astype(str).str.extract(r"^([A-Z]{2,3}|[A-Z]\d|\d[A-Z])", expand=False).fillna("?")
d["mon"] = d.month.astype(str)
d = d[np.isfinite(d.y) & (d.y > 0)]
print(f"LIRF no-record departures: {len(d)} | echo {d.echo.mean():.3f}")
print("\n== echo rate by month ==")
print(d.groupby("mon").echo.agg(["size", "mean"]).round(3).T.to_string())
top = d.carrier.value_counts()
print("\n== carriers (>= 15 rows): echo rate overall, Jan, Jul, and by quarter ==")
d["q"] = pd.to_datetime(d.mon).dt.quarter
rows = []
for c, n in top.items():
    if n < 15: continue
    g = d[d.carrier == c]
    rows.append({"carrier": c, "n": n, "echo": g.echo.mean(),
                 **{f"Q{q}": g.echo[g.q == q].mean() for q in (1, 2, 3, 4)},
                 "Jan": g.echo[g.mon == "2025-01"].mean(), "Jul": g.echo[g.mon == "2025-07"].mean(),
                 "y_med_echo": g.y[g.echo].median(), "y_med_noecho": g.y[~g.echo].median()})
print(pd.DataFrame(rows).round(2).to_string(index=False))
# honest predictive value: leave-one-month-out carrier echo rate (smoothed toward the LIRF mean)
k = 10.0
pred = np.full(len(d), np.nan)
for m in d.mon.unique():
    tr, te = d[d.mon != m], d.mon == m
    base = tr.echo.mean()
    st = tr.groupby("carrier").echo.agg(["sum", "size"])
    rate = (st["sum"] + k * base) / (st["size"] + k)
    pred[te.to_numpy()] = d.loc[te, "carrier"].map(rate).fillna(base).to_numpy()
d["p_echo"] = pred
lomo_base = np.array([d[d.mon != m].echo.mean() for m in d.mon])
print(f"\nLOMO Brier: LIRF-constant {np.mean((d.echo - lomo_base)**2):.4f} vs carrier-smoothed {np.mean((d.echo - d.p_echo)**2):.4f}")
for name, m in (("Jan", d.mon == "2025-01"), ("Jul", d.mon == "2025-07")):
    print(f"  {name}: constant {np.mean((d.echo[m] - lomo_base[m.to_numpy()])**2):.4f} vs carrier {np.mean((d.echo[m] - d.p_echo[m])**2):.4f} (n={m.sum()})")
d[["carrier", "mon", "ds", "y", "echo", "p_echo"]].to_parquet(Path(__file__).parent / "lirf_norec_lomo.parquet")
