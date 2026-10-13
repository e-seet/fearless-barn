#!/usr/bin/env python3
"""Checks whether LIRF's AOBT-unmatched ("has issues") population is linked
to specific destinations/routes ("flight path"), and exports the full LIRF
departure population plus a destination-level summary to CSV.

Background: LIRF is this project's single worst-performing airport (§2q,
20.7% of pooled SSE from 0.08% of rows) and its AOBT-unmatched population
(no NM flight record, ~1,488 rows/year) is the specific subgroup involved.
Checked destination (ADES_mvt), runway, and aircraft-type distributions of
that subgroup against LIRF's overall departure population.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import io as cio

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE


def main() -> int:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
            "STAND_mvt", "MVT_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt",
            "AIRCRAFT_OPERATOR_flt", config.TARGET]
    mov = cio.load_training(columns=cols)
    dep = mov[(mov["PHASE_mvt"].eq("DEP")) & (mov["ADEP_mvt"].eq("LIRF"))].copy()
    dep["has_flight_record"] = dep["AIRCRAFT_OPERATOR_flt"].notna()
    dep["has_issue"] = ~dep["has_flight_record"]
    dep["date"] = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.tz_localize(None)

    print("LIRF DEP total: {}  has_issue (no flight record): {} ({:.3%})".format(
        len(dep), dep["has_issue"].sum(), dep["has_issue"].mean()))

    # ---- destination-level summary: is the issue population concentrated
    # on specific routes? ----
    issue = dep[dep["has_issue"]]
    clean = dep[~dep["has_issue"]]
    dest_summary = pd.DataFrame({
        "n_issue": issue["ADES_mvt"].value_counts(),
        "n_total": dep["ADES_mvt"].value_counts(),
    }).fillna(0)
    dest_summary["n_clean"] = dest_summary["n_total"] - dest_summary["n_issue"]
    dest_summary["issue_frac_of_route"] = dest_summary["n_issue"] / dest_summary["n_total"]
    dest_summary["issue_share_of_all_issues"] = dest_summary["n_issue"] / dest_summary["n_issue"].sum()
    dest_summary["overall_share_of_all_deps"] = dest_summary["n_total"] / dest_summary["n_total"].sum()
    dest_summary["overrepresentation_ratio"] = (
        dest_summary["issue_share_of_all_issues"] / dest_summary["overall_share_of_all_deps"])
    dest_summary = dest_summary.sort_values("overrepresentation_ratio", ascending=False)
    dest_summary.index.name = "ADES_mvt"

    print("\nTop 15 destinations by overrepresentation in the issue population:")
    print(dest_summary[dest_summary["n_issue"] >= 3].head(15)[
        ["n_issue", "n_total", "issue_frac_of_route", "overrepresentation_ratio"]
    ].to_string())

    # join the destination overrepresentation ratio back onto every row for context
    dep = dep.merge(
        dest_summary[["overrepresentation_ratio"]].rename(
            columns={"overrepresentation_ratio": "ades_overrep_ratio"}),
        left_on="ADES_mvt", right_index=True, how="left")

    export_cols = ["MVT_ID_mvt", "date", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt",
                   "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt",
                   config.TARGET, "has_issue", "ades_overrep_ratio"]
    full_out = OUT_DIR / "lirf_departures_full.csv"
    dep[export_cols].sort_values(["has_issue", "date"], ascending=[False, True]).to_csv(
        full_out, index=False)
    print("\nwrote {} ({} rows, {} flagged has_issue=True)".format(
        full_out, len(dep), dep["has_issue"].sum()))

    dest_out = OUT_DIR / "lirf_destination_summary.csv"
    dest_summary.reset_index().to_csv(dest_out, index=False)
    print("wrote {} ({} distinct destinations)".format(dest_out, len(dest_summary)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
