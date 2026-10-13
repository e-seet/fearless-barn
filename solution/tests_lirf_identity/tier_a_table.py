#!/usr/bin/env python3
"""Consolidated results for every Tier-A arm (see ../pipeline_review.csv).
Reads eval_{val,test}.json across all seeds on disk; reports mean and range."""
from __future__ import annotations
import json, statistics as st
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "artifacts" / "runs"
PROTO = "forward-with_aobt-keep_all"

# (label, arm, tag)
ARMS = [
    ("multimodel_antigravity  (SHIP)", "multimodel_antigravity", ""),
    ("cross_blend w_inner  (tie)",     "cross_blend",            "w_inner"),
    ("airport_operator_blend",         "airport_operator_blend", ""),
    ("per_airport",                    "per_airport",            ""),
    ("lightgbm  (reference)",          "lightgbm",               ""),
    ("ridge  (linear floor)",          "ridge",                  ""),
    ("naive_offblock  (trivial floor)","naive_offblock",         ""),
    ("lirf_identity idband  (add-on)", "airport_lirf_identity_multimodel", "idband"),
    ("lirf_identity FULL  (add-on+bet)","airport_lirf_identity_multimodel", ""),
]
KEYS = [("rmse", "val"), ("rmse_in_band", "val"),
        ("rmse", "test"), ("rmse_in_band", "test"),
        ("leaderboard_weighted_rmse", "test")]
HDR = ["val", "val_ib", "test", "test_ib", "test_LBw"]


def collect(arm, tag):
    suffix = f"__tag={tag}" if tag else ""
    out = {}
    for d in sorted(RUNS.glob(f"{arm}__{PROTO}__seed*{suffix}")):
        if tag == "" and "__tag=" in d.name:
            continue
        seed = int(d.name.split("__seed")[1].split("__")[0])
        ev = {p: json.loads((d / f"eval_{p}.json").read_text()) for p in ("val", "test")
              if (d / f"eval_{p}.json").exists()}
        if "val" in ev and "test" in ev:
            out[seed] = ev
    return out


def fmt(vals):
    if not vals:
        return "        --"
    if len(vals) == 1:
        return f"{vals[0]:9.2f}"
    return f"{st.mean(vals):9.2f}"


def rng(vals):
    return "" if len(vals) < 2 else f"  (n={len(vals)}, range {max(vals)-min(vals):.2f})"


print(f"\nTier-A arms  --  forward / with_aobt / keep_all  --  mean over seeds on disk\n")
print(f"{'arm':34s}" + "".join(f"{h:>10s}" for h in HDR) + "   seeds")
print("-" * 104)
for label, arm, tag in ARMS:
    runs = collect(arm, tag)
    seeds = sorted(runs)
    cols = []
    for key, part in KEYS:
        vals = [runs[s][part].get(key) for s in seeds if runs[s][part].get(key) is not None]
        cols.append(fmt(vals))
    seedstr = "/".join(str(s) for s in seeds) if seeds else "none"
    print(f"{label:34s}" + "".join(cols) + f"   {seedstr}")
    # ranges on the low-variance metric
    tib = [runs[s]["test"].get("rmse_in_band") for s in seeds if runs[s]["test"].get("rmse_in_band")]
    vib = [runs[s]["val"].get("rmse_in_band") for s in seeds if runs[s]["val"].get("rmse_in_band")]
    if len(seeds) > 1:
        print(f"{'':34s}{'':>10s}{'':>10s}{'':>10s}"
              f"  val_ib{rng(vib)}   test_ib{rng(tib)}")
print()
