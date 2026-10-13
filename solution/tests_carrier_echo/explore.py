#!/usr/bin/env python3
"""Carrier-echo study, step 1 (read-only): for departures with no NM flight record, does the
carrier predict whether the recorded off-block copies the scheduled time ("echo")?"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from common import io as cio
pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 80)
r = cio.load_training()
d = r[r.PHASE_mvt.eq("DEP")].copy()
t = lambda c: pd.to_datetime(d[c], utc=True, errors="coerce")
d["y"] = pd.to_numeric(d.TAXITIME_SEC_mvt, errors="coerce")
d["ds"] = (t("MVT_TIME_UTC_mvt") - t("SCHED_TIME_UTC_mvt")).dt.total_seconds()
d["block_minus_sched"] = (t("BLOCK_TIME_UTC_mvt") - t("SCHED_TIME_UTC_mvt")).dt.total_seconds()
d["echo"] = d.block_minus_sched.abs() <= 60
d["norec"] = d.FLIGHT_ID_mvt.isna()
d["carrier"] = d.FLIGHT_mvt.astype(str).str.extract(r"^([A-Z]{2,3}|[A-Z]\d|\d[A-Z])", expand=False).fillna("?")
n = d[d.norec & np.isfinite(d.y) & (d.y > 0)]
print(f"no-record departures: {len(n)} of {len(d)} | echo rate overall {n.echo.mean():.3f}")
print("\n== by airport: rows, echo rate, echo rate when ds > 1 h ==")
print(n.groupby("ADEP_mvt").apply(lambda g: pd.Series({"n": len(g), "echo": g.echo.mean(),
      "n_ds>1h": int((g.ds > 3600).sum()), "echo|ds>1h": g.echo[g.ds > 3600].mean()})).round(3).to_string())
big = n[n.ds > 3600]
print(f"\n== no-record, ds > 1 h: {len(big)} rows; share of their squared error if predicted 1,000 s: "
      f"{((big.y-1000)**2).sum()/((n.y-1000)**2).sum():.1%} of all no-record SE ==")
top = big.carrier.value_counts().head(25).index
g = big[big.carrier.isin(top)].groupby("carrier").apply(lambda g: pd.Series({
    "n": len(g), "airports": ",".join(sorted(g.ADEP_mvt.unique())[:4]), "echo": g.echo.mean(),
    "echo_H1": g.echo[g.month.astype(str) <= "2025-06"].mean(), "echo_H2": g.echo[g.month.astype(str) > "2025-06"].mean(),
    "y_med": g.y.median(), "y_mean": g.y.mean()})).sort_values("n", ascending=False)
print("\n== top carriers among no-record, ds > 1 h (echo rate, split by half-year for stability) ==")
print(g.round(3).to_string())
# how much of the echo variance does carrier explain (ds > 1 h rows)?
cr = big.groupby("carrier").echo.transform("mean")
base = big.echo.mean()
print(f"\nBrier: constant {np.mean((big.echo-base)**2):.4f} vs carrier mean (in-sample) {np.mean((big.echo-cr)**2):.4f}")
