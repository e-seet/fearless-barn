#!/usr/bin/env python3
"""Build fearless_barn_v28.parquet: v25's shipped predictions, UNCHANGED,
except for LIRF's AOBT-unmatched rows (ADEP_mvt == "LIRF" and AOBT_3_flt is
null -- 383 of 344,841 real ranking rows, 0.111%), where the prediction is
replaced with the plain schedule proxy `delta_sched_sec = MVT_TIME - SCHED_TIME`.

This is a diagnostic PROBE, not a candidate for best (same use of the word
as v27, TAXI_OUT_DRIVERS.md sec.2ai). Motivation and local numbers are in
`estimate_local_effect.py`'s output and the chat writeup: this population
carries outsized SSE leverage (0.077-0.11% of rows, ~20% of SSE locally) so
even a small board move here is informative, unlike this project's usual
sub-5s closed features. Local signal is genuinely mixed across surfaces
(forward val: proxy +13.33s worse pooled; forward test: proxy -2.61s
better pooled) -- submitting this tells us whether the real board agrees
with either local read, which bears on the open question of whether our
board gap vs. a leaderboard competitor (sec.2al) is LIRF-specific.

Does NOT upload. Mirrors submit.py's own validate() checks and prints the
same manual-upload instructions, so sending stays an explicit human step.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import io as cio

BASE_VERSION = 25
NEW_VERSION = 28
SUBMIT_DIR = config.SOLUTION_DIR / "submissions"


def validate(sub: "pd.DataFrame") -> "list[str]":
    """Identical to submit.py's validate() -- kept in sync by hand since this
    script builds a submission file without going through submit.py's own
    predict_ranking() path."""
    tmpl = pd.read_parquet(config.SUBMITTING_PARQUET)
    errs = []
    if list(sub.columns) != [config.ID, config.TARGET]:
        errs.append("columns are {}, must be exactly ['{}', '{}']".format(
            list(sub.columns), config.ID, config.TARGET))
    if len(sub) != len(tmpl):
        errs.append("row count {} != template {}".format(len(sub), len(tmpl)))
    missing = set(tmpl[config.ID]) - set(sub[config.ID])
    extra = set(sub[config.ID]) - set(tmpl[config.ID])
    if missing:
        errs.append("{} MVT_ID present in template but missing from submission".format(len(missing)))
    if extra:
        errs.append("{} MVT_ID in submission but not in template".format(len(extra)))
    if sub[config.ID].duplicated().any():
        errs.append("{} duplicated MVT_ID".format(int(sub[config.ID].duplicated().sum())))
    n_bad = int((~np.isfinite(sub[config.TARGET].astype("float64"))).sum())
    if n_bad:
        errs.append("{} non-finite predictions (NaN/inf) -- these score as failures".format(n_bad))
    neg = int((sub[config.TARGET] < 0).sum())
    if neg:
        errs.append("{} negative taxi-time predictions".format(neg))
    return errs


def main() -> int:
    base_path = SUBMIT_DIR / "fearless_barn_v{}.parquet".format(BASE_VERSION)
    base = pd.read_parquet(base_path)
    print("[v28-probe] loaded base {} ({} rows)".format(base_path, len(base)))

    rank = cio.load_ranking(columns=[
        "MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "AOBT_3_flt",
        "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    dep = rank[rank["PHASE_mvt"].eq("DEP")].copy()
    dep["MVT_TIME_UTC_mvt"] = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True).dt.tz_localize(None)
    dep["SCHED_TIME_UTC_mvt"] = pd.to_datetime(dep["SCHED_TIME_UTC_mvt"], utc=True).dt.tz_localize(None)
    dep["delta_sched_sec"] = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()

    target_mask = dep["ADEP_mvt"].eq("LIRF") & dep["AOBT_3_flt"].isna()
    targets = dep.loc[target_mask, ["MVT_ID_mvt", "delta_sched_sec"]]
    print("[v28-probe] override population: {} rows ({:.4%} of {} DEP rows)".format(
        len(targets), len(targets) / len(dep), len(dep)))

    sub = base.copy()
    proxy_by_id = dict(zip(targets["MVT_ID_mvt"], targets["delta_sched_sec"]))
    override_ids = set(proxy_by_id)
    n_overridden = sub[config.ID].isin(override_ids).sum()
    assert n_overridden == len(targets), (
        "ID mismatch: {} rows in submission match the override population, "
        "expected {}".format(n_overridden, len(targets)))

    proxy_vals = sub[config.ID].map(proxy_by_id)
    proxy_clipped = np.clip(proxy_vals, config.PREDICTION_CLIP_MIN_SEC,
                             config.PREDICTION_CLIP_MAX_SEC)
    before = sub.loc[sub[config.ID].isin(override_ids), config.TARGET].to_numpy()
    sub.loc[sub[config.ID].isin(override_ids), config.TARGET] = (
        np.rint(proxy_clipped[sub[config.ID].isin(override_ids)]).astype("int32"))
    after = sub.loc[sub[config.ID].isin(override_ids), config.TARGET].to_numpy()

    print("[v28-probe] overridden {} rows -- model pred vs proxy pred (sample):".format(n_overridden))
    print("  before (model):  min={} p50={} max={}".format(before.min(), np.median(before), before.max()))
    print("  after  (proxy):  min={} p50={} max={}".format(after.min(), np.median(after), after.max()))
    print("  every other row unchanged: {}".format(
        (sub.loc[~sub[config.ID].isin(override_ids), config.TARGET].to_numpy()
         == base.loc[~base[config.ID].isin(override_ids), config.TARGET].to_numpy()).all()))

    print("\n== submission validation ==")
    errs = validate(sub)
    if errs:
        print("  REJECTED:")
        for e in errs:
            print("    - " + e)
        return 1
    print("  [PASS] columns, row count, ID set, uniqueness, finiteness, sign")

    name = "fearless_barn_v{}.parquet".format(NEW_VERSION)
    path = SUBMIT_DIR / name
    sub.to_parquet(path, index=False)
    (SUBMIT_DIR / (name + ".meta.json")).write_text(json.dumps({
        "kind": "diagnostic_probe",
        "base_version": BASE_VERSION,
        "override": "LIRF AOBT-unmatched rows -> delta_sched_sec proxy",
        "override_n_rows": int(n_overridden),
        "override_share": float(n_overridden / len(sub)),
        "team": config.TEAM,
        "version": NEW_VERSION,
        "n_rows": int(len(sub)),
    }, indent=2))
    print("\n  wrote {}".format(path))
    team_hyphen = config.TEAM.replace("_", "-")
    print("\n== upload (run this yourself -- this script does not upload) ==")
    print("  mc cp \"submissions\\{}\" \"dc26/prc-2026-{}/{}_v{}.parquet\"".format(
        name, team_hyphen, team_hyphen, NEW_VERSION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
