#!/usr/bin/env python3
"""Seed-robustness analysis for the airport_*_multimodel family.

Question under test (user's, 2026-09-07):
  Is the ranking that crowns `airport_wtc_rec_multimodel` / `airport_unsup_multimodel`
  as the "3-seed champion" a real effect, or is it noise-mining -- i.e. are the
  win margins between the sibling arms smaller than each arm's own seed spread?

Reads eval_{val,test}.json off disk for every seed present. Re-run after new seeds
land to extend the table.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "runs"
PROTO = "forward-with_aobt-keep_all"

ARMS = [
    "airport_unsup_multimodel",
    "airport_wtc_rec_multimodel",
    "airport_wtc_multimodel",
    "airport_operator_multimodel",
    "airport_runway_multimodel",
    "multimodel_antigravity",
]
SEEDS = [42, 43, 44, 45, 46, 47]


def load(arm, seed, part):
    f = RUNS / f"{arm}__{PROTO}__seed{seed}" / f"eval_{part}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d["rmse"], d["rmse_in_band"]


def col(rows, arm, key):
    return [r[key] for r in rows if r["arm"] == arm and r[key] is not None]


def main():
    rows = []
    for arm in ARMS:
        for s in SEEDS:
            v = load(arm, s, "val")
            t = load(arm, s, "test")
            if v is None and t is None:
                continue
            rows.append({
                "arm": arm, "seed": s,
                "val": v[0] if v else None, "val_ib": v[1] if v else None,
                "test": t[0] if t else None, "test_ib": t[1] if t else None,
            })

    seeds_present = sorted({r["seed"] for r in rows})
    print(f"seeds on disk: {seeds_present}\n")

    for metric in ("val", "test", "val_ib", "test_ib"):
        print(f"================  {metric.upper()}  ================")
        hdr = f"{'arm':30s}" + "".join(f"  s{s:<7d}" for s in seeds_present) + \
              f"  {'mean':>8s} {'sd':>6s} {'range':>6s}"
        print(hdr)
        summary = []
        for arm in ARMS:
            vals = {r["seed"]: r[metric] for r in rows if r["arm"] == arm}
            series = [vals[s] for s in seeds_present if s in vals and vals[s] is not None]
            if not series:
                continue
            cells = "".join(
                (f"  {vals[s]:8.2f}" if s in vals and vals[s] is not None else f"  {'--':>8s}")
                for s in seeds_present
            )
            mean = st.mean(series)
            sd = st.pstdev(series) if len(series) > 1 else 0.0
            rng = (max(series) - min(series)) if len(series) > 1 else 0.0
            print(f"{arm:30s}{cells}  {mean:8.2f} {sd:6.2f} {rng:6.2f}")
            summary.append((arm, mean, sd, rng, series))
        # ranking per seed
        print("\n  ranking by seed (best first):")
        for s in seeds_present:
            order = sorted(
                [(r["arm"], r[metric]) for r in rows if r["seed"] == s and r[metric] is not None],
                key=lambda x: x[1],
            )
            if order:
                print(f"    s{s}: " + "  <  ".join(f"{a.replace('airport_','').replace('_multimodel','')}({v:.2f})" for a, v in order))
        # margin vs noise
        if len(summary) >= 2:
            summary.sort(key=lambda x: x[1])
            best_arm, best_mean, best_sd, _, _ = summary[0]
            run_arm, run_mean, run_sd, _, _ = summary[1]
            gap = run_mean - best_mean
            noise = max(best_sd, run_sd)
            print(f"\n  #1 {best_arm.replace('airport_','')} mean {best_mean:.2f}  vs  "
                  f"#2 {run_arm.replace('airport_','')} mean {run_mean:.2f}")
            print(f"  win margin = {gap:.2f} s   |   larger of the two arms' seed SD = {noise:.2f} s"
                  f"   |   margin/noise = {gap / noise:.2f}" if noise else "")
            spread = summary[-1][1] - summary[0][1]
            print(f"  full family spread (worst mean - best mean) = {spread:.2f} s")
        print()


if __name__ == "__main__":
    main()
