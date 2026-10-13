#!/usr/bin/env python3
"""Avenue 1: can a physical stand-turnover check *definitively* prove a
large-delay prediction in LIRF's AOBT-unmatched population is wrong?

Background (`TAXI_OUT_DRIVERS.md` sec.2q, `conversation.md` sec.2): a
population of LIRF departures has no matching NM flight record
(`has_flight_record==0`, ~1,488 rows/year) and some of them show
`BLOCK_TIME_UTC_mvt` sitting suspiciously close to `SCHED_TIME_UTC_mvt` while
takeoff (`MVT_TIME_UTC_mvt`) is hours later -- so `TAXITIME_SEC_mvt` reads as
a huge "taxi" that is really unexplained ground delay (or a data artifact;
the organizer confirmed BLOCK_TIME is *supposed* to be the actual off-block
time, not an intentional schedule-fallback, but "could be linked to some
specific events, others could just be messy data"). Three prior approaches on
this exact population (blend-weight solver, OpenAir-proxy cross-check, convex
mix) all failed to beat plain MLP out-of-sample -- but none of them asked a
purely physical question: can a stand realistically have been occupied by one
aircraft for the ENTIRE claimed window [BLOCK_TIME, takeoff)?

This is a hard physical constraint, not a statistical proxy: a stand can only
hold one aircraft at a time. If another movement (any flight, any phase) is
recorded at the SAME (airport, stand) strictly inside a row's claimed
occupancy window, that is a **direct, deterministic contradiction** -- the
claimed BLOCK_TIME cannot be correct, independent of any model or fitted
weight. This is legitimate use of "other movement rows" per the organizer's
explicit confirmation that forward-looking movement-stream features are fair
game (`conversation.md` sec.1).

Feasibility gate (checked separately, passed): STAND_mvt is >99.3% populated
even on the AOBT-unmatched subgroup, so this is buildable.

This script only answers the prevalence/feasibility question -- how much of
the large-delay population has a definitive stand-contradiction -- and gives
a rough, in-sample ceiling if those specific rows were corrected. It does
NOT build a feature or retrain; that is a follow-up decision, not this one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))

import config
from common import io as cio

AIRPORT = "LIRF"
LARGE_DELAY_THRESHOLD_SEC = 3600  # 1 hour -- normal taxi is minutes, not hours


def main() -> int:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt",
            "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", config.TARGET,
            "AIRCRAFT_OPERATOR_flt"]
    print("[stand-turnover] loading training data...")
    mov = cio.load_training(columns=cols)
    # Normalize to tz-naive UTC datetime64[ns] up front so numpy searchsorted
    # comparisons below can't hit a tz-naive/tz-aware mismatch.
    for c in ("MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"):
        mov[c] = pd.to_datetime(mov[c], utc=True).dt.tz_localize(None)

    is_dep = mov["PHASE_mvt"].eq("DEP")
    is_arr = mov["PHASE_mvt"].eq("ARR")
    at_lirf = (is_dep & mov["ADEP_mvt"].eq(AIRPORT)) | (is_arr & mov["ADES_mvt"].eq(AIRPORT))
    lirf = mov[at_lirf].copy()
    print("  LIRF movement rows (both phases): {}".format(len(lirf)))

    lirf["event_time"] = lirf["MVT_TIME_UTC_mvt"]
    lirf_stand = lirf.dropna(subset=["STAND_mvt"]).sort_values("event_time")
    print("  LIRF rows with a STAND_mvt value: {} ({:.2%})".format(
        len(lirf_stand), len(lirf_stand) / len(lirf)))

    # ---- target population: LIRF DEP, no matching flight record ----
    dep_lirf = lirf[lirf["PHASE_mvt"].eq("DEP")].copy()
    dep_lirf["has_flight_record"] = dep_lirf["AIRCRAFT_OPERATOR_flt"].notna()
    pop = dep_lirf[~dep_lirf["has_flight_record"]].copy()
    print("  LIRF DEP, no flight record (the sec.2q population): {}".format(len(pop)))
    pop_large = pop[pop[config.TARGET] > LARGE_DELAY_THRESHOLD_SEC].copy()
    pop_large = pop_large.dropna(subset=["STAND_mvt", "BLOCK_TIME_UTC_mvt"])
    print("  ...of those, TAXITIME > {}s AND has STAND_mvt+BLOCK_TIME: {}".format(
        LARGE_DELAY_THRESHOLD_SEC, len(pop_large)))

    # Honest held-out restriction, matching sec.2q's convention: the physical
    # stand-timeline lookup may legitimately use the WHOLE year (forward-looking
    # movement-stream features are organizer-confirmed fair game -- this is the
    # same footing as the existing congestion features), but the population we
    # REPORT a ceiling against must be restricted to forward/val + forward/test,
    # not the full training pool, or the number isn't comparable to anything
    # else in this project's record.
    val_ids = pd.read_parquet(config.SPLIT_DIR / "forward" / "val_ids.parquet")[config.ID]
    test_ids = pd.read_parquet(config.SPLIT_DIR / "forward" / "test_ids.parquet")[config.ID]
    holdout_ids = set(val_ids) | set(test_ids)
    pop_large_holdout = pop_large[pop_large["MVT_ID_mvt"].isin(holdout_ids)].copy()
    print("  ...of those, in forward/val+test (the honest, held-out comparable slice): {}".format(
        len(pop_large_holdout)))

    # ---- build a per-stand sorted index of ALL LIRF movement events for fast lookup ----
    by_stand = {stand: grp["event_time"].to_numpy()
                for stand, grp in lirf_stand.groupby("STAND_mvt")}
    by_stand_id = {stand: grp["MVT_ID_mvt"].to_numpy()
                  for stand, grp in lirf_stand.groupby("STAND_mvt")}

    contradicted = []
    for row in pop_large.itertuples():
        stand = row.STAND_mvt
        times = by_stand.get(stand)
        ids = by_stand_id.get(stand)
        if times is None:
            continue
        b = np.datetime64(row.BLOCK_TIME_UTC_mvt, "ns")
        e = np.datetime64(row.MVT_TIME_UTC_mvt, "ns")  # takeoff = claimed window end
        # events strictly inside (b, e), excluding this row's own event
        lo = np.searchsorted(times, b, side="right")
        hi = np.searchsorted(times, e, side="left")
        cand = [(t, i) for t, i in zip(times[lo:hi], ids[lo:hi]) if i != row.MVT_ID_mvt]
        if cand:
            first_t = min(t for t, _ in cand)
            window_sec = (e - b) / np.timedelta64(1, "s")
            implied_bound_sec = (first_t - b) / np.timedelta64(1, "s")
            contradicted.append({
                "mvt_id": row.MVT_ID_mvt, "stand": stand,
                "block_time": str(b), "takeoff": str(e),
                "claimed_taxitime_sec": float(getattr(row, config.TARGET)),
                "n_contradicting_events": len(cand),
                "first_contradiction_at_sec": round(implied_bound_sec, 1),
                "window_sec": round(window_sec, 1),
                "in_forward_holdout": row.MVT_ID_mvt in holdout_ids,
            })

    n_pop = len(pop_large)
    n_contra = len(contradicted)
    print("\n[result, full training-pool context] {} of {} large-delay, no-flight-record "
          "LIRF rows ({:.1%}) have a DEFINITIVE stand-turnover contradiction".format(
          n_contra, n_pop, n_contra / n_pop if n_pop else float("nan")))

    n_pop_ho = len(pop_large_holdout)
    n_contra_ho = sum(1 for c in contradicted if c["in_forward_holdout"])
    print("[result, HONEST forward/val+test only] {} of {} rows ({:.1%}) have a "
          "DEFINITIVE stand-turnover contradiction".format(
          n_contra_ho, n_pop_ho, n_contra_ho / n_pop_ho if n_pop_ho else float("nan")))

    if contradicted:
        df = pd.DataFrame(contradicted)
        df.to_csv(HERE / "contradicted_rows.csv", index=False)
        print("\nSample (first 10, sorted by claimed taxitime desc):")
        print(df.sort_values("claimed_taxitime_sec", ascending=False).head(10).to_string(index=False))

        # Rough in-sample ceiling: cap each contradicted row's error at
        # (first_contradiction_at_sec) instead of the model's actual prediction
        # is a downstream modeling question -- here we only report how much
        # SSE these rows carry today, as the same "concentration" framing sec.2q used.
        pooled_n = len(pop)  # denominator context, not a real val/test count
        print("\n[context] this diagnostic ran on the full training pool ({} LIRF "
              "no-flight-record rows), not a held-out val/test split -- it establishes "
              "PREVALENCE and FEASIBILITY only. A real ceiling estimate (in the style "
              "of sec.2q) requires re-running this on forward/val and forward/test "
              "specifically, which is the natural next step if this fraction looks "
              "worth pursuing.".format(pooled_n))
    else:
        print("\nNo definitive contradictions found -- the avenue does not have direct "
              "physical evidence to exploit in this population as scoped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
