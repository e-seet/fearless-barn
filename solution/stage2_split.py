#!/usr/bin/env python3
"""Stage 2 -- build and audit the temporal splits.

Contract
  Trigger : stage1_audit.py passed
  Action  : materialise the month-based train/val/test MVT_ID manifests for a
            split protocol, then re-audit the split THIS stage just wrote
  Output  : artifacts/splits/<protocol>/{train,val,test}_ids.parquet + split_report.json
  Gate    : the three ID sets are pairwise disjoint; no month appears in two
            sets; every set is non-empty; the airport filter was applied

Leak gate placement (pipeline-testing.md): the stage that CREATES a split
re-audits its own output. Not deferred to a downstream consumer.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import io as cio
from common.bootstrap import assert_split_disjoint, write_json_guarded


def build_split(split: str):
    spec = config.SPLITS[split]
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"]
    df = cio.load_training(columns=cols)
    dep = df[df.PHASE_mvt.eq("DEP")]

    assigned = {}
    for part in ("train", "val", "test"):
        months = spec[part]
        if not months:          # the 'full' refit protocol has no val/test
            assigned[part] = dep.iloc[0:0]
            continue
        sel = dep[dep.month.isin(months)]
        filt = spec.get("airport_filter") or {}
        if filt:
            keep = pd.Series(True, index=sel.index)
            for m, airports in filt.items():
                if m in months:
                    in_m = sel.month.eq(m)
                    keep &= ~in_m | sel.ADEP_mvt.isin(airports)
            sel = sel[keep]
        assigned[part] = sel
    return assigned, spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=config.DEFAULT_SPLIT, choices=sorted(config.SPLITS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    assigned, spec = build_split(args.split)
    outdir = config.SPLIT_DIR / args.split
    outdir.mkdir(parents=True, exist_ok=True)

    print("\n== split '{}' ==".format(args.split))
    for part, sel in assigned.items():
        print("  {:5s} n={:8d} months={} airports={}".format(
            part, len(sel), spec[part], sorted(sel.ADEP_mvt.unique())))
        sel[["MVT_ID_mvt", "ADEP_mvt", "month"]].to_parquet(
            outdir / "{}_ids.parquet".format(part), index=False)

    # ---- gates ----
    print("\n== split gates ==")
    failures = []
    ids = {p: set(s.MVT_ID_mvt) for p, s in assigned.items()}
    # ml_common's assert_split_disjoint rejects an empty part by design (an empty
    # split is normally a bug). The 'full' refit protocol legitimately has none,
    # so pass it only the populated parts -- and say so, rather than skipping the
    # gate silently.
    populated = {p: sorted(v) for p, v in ids.items() if v}
    if len(populated) < len(ids):
        print("  [NOTE] disjointness checked over {} populated part(s); {} "
              "intentionally empty (refit-only protocol)".format(
                  sorted(populated), sorted(set(ids) - set(populated))))
    dj = (assert_split_disjoint(populated) if len(populated) > 1
          else {"n_parts": len(populated), "note": "single populated part; "
                "disjointness is vacuous"})
    print("  [PASS] pairwise ID disjointness ({} ids total)".format(
        sum(len(v) for v in ids.values())))

    months_used = {p: set(assigned[p].month.unique()) for p in assigned}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b and months_used[a] & months_used[b]:
                failures.append("months shared between {} and {}: {}".format(
                    a, b, sorted(months_used[a] & months_used[b])))
    for p, s in assigned.items():
        if s.empty and spec[p]:
            failures.append("{} split is empty but names months {}".format(p, spec[p]))
    if not assigned["train"].empty and not spec["val"] and not spec["test"]:
        print("  [NOTE] '{}' is a refit-only protocol: no val/test, so nothing may "
              "be SELECTED on it -- hyperparameters must arrive frozen".format(args.split))
    filt = spec.get("airport_filter") or {}
    for m, airports in filt.items():
        for p, s in assigned.items():
            bad = s[s.month.eq(m) & ~s.ADEP_mvt.isin(airports)]
            if len(bad):
                failures.append("airport filter not applied for {} in {}: {} rows".format(
                    m, p, len(bad)))
    if not failures:
        print("  [PASS] month exclusivity, non-empty sets, airport filter applied")

    rep = {
        "split": args.split, "spec": spec,
        "sizes": {p: int(len(s)) for p, s in assigned.items()},
        "months": {p: sorted(months_used[p]) for p in assigned},
        "airports": {p: sorted(assigned[p].ADEP_mvt.unique().tolist()) for p in assigned},
        "disjointness": dj, "gates_failed": failures, "passed": not failures,
    }
    write_json_guarded(outdir / "split_report.json", rep, force=args.force)

    if failures:
        print("\n  SPLIT GATE FAILED:")
        for f in failures:
            print("    - " + f)
        return 1
    print("\n  all split gates PASS -> {}".format(outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
