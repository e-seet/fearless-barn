#!/usr/bin/env python3
"""Diagnostic (no training): pooled matched-vs-unmatched RMSE split for this
project's own best shipped-style arm, mirroring a competitor's ledger.

Motivation: reviewing `saurabhdxt-labs/prc-challenge` (team merry-quicksand,
rank 7/160)'s `reports/MSE_LEDGER.md` found they track

    RMSE^2 = w_matched * matched_RMSE^2 + w_unmatched * unmatched_RMSE^2

(matched = AOBT_3_flt present, ~98.5% of rows; unmatched = no NM off-block
record at all, ~1.5%) as their central bookkeeping identity, and report
matched RMSE ~226-237 / unmatched (their "stratum") RMSE ~1868-1946 on their
own held-out fold. This project has quantified the unmatched population's
severity for LIRF specifically (SS2q: 283 rows, RMSE 4018.88, 20.68% of SSE)
but never computed this pooled, all-10-airport split the way their ledger
does -- which is the natural apples-to-apples comparison for asking whether
OUR local-to-board gap (a well-documented +47 to +51s, opposite in sign from
their early-submission -28.5s) concentrates in the matched lane, the
unmatched lane, or both.

Method: `airport_operator_multimodel` (tag=msmlp, seed42), the same
architecture family as the actual shipped v25 submission, on `forward`
val AND test (the two surfaces this project's other diagnostics use).
`has_aobt` (already a built feature) is the matched/unmatched split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import config
from common import dataset as D
from common.predict import predict_arm

SPLIT, FEATURESET, OUTLIER_POLICY = "forward", "with_aobt", "keep_all"
ARM, SEED, TAG = "airport_operator_multimodel", 42, "msmlp"


def rmse(err):
    return float(np.sqrt(np.mean(err ** 2)))


def check(part: str) -> None:
    print("\n" + "=" * 70)
    print("[matched-unmatched] part = {}".format(part))
    X, y, meta = D.load_part(SPLIT, part, FEATURESET, filter_target=False)
    y = np.asarray(y, dtype="float64")
    p = predict_arm(ARM, SPLIT, part, FEATURESET, SEED, OUTLIER_POLICY,
                     X=X, meta=meta, tag=TAG)
    err = y - p

    matched = X["has_aobt"].to_numpy().astype(bool)
    n = len(y)
    w_matched = matched.mean()
    w_unmatched = 1.0 - w_matched
    rmse_matched = rmse(err[matched])
    rmse_unmatched = rmse(err[~matched])
    rmse_pooled_direct = rmse(err)
    rmse_pooled_reconstructed = float(np.sqrt(
        w_matched * rmse_matched ** 2 + w_unmatched * rmse_unmatched ** 2))

    print("  n={}  matched={} ({:.4%})  unmatched={} ({:.4%})".format(
        n, matched.sum(), w_matched, (~matched).sum(), w_unmatched))
    print("  RMSE matched   = {:.2f}".format(rmse_matched))
    print("  RMSE unmatched = {:.2f}".format(rmse_unmatched))
    print("  RMSE pooled (direct)        = {:.3f}".format(rmse_pooled_direct))
    print("  RMSE pooled (reconstructed) = {:.3f}  (sanity check, should match)".format(
        rmse_pooled_reconstructed))
    print("  SSE share: matched {:.1%}  unmatched {:.1%}".format(
        np.sum(err[matched] ** 2) / np.sum(err ** 2),
        np.sum(err[~matched] ** 2) / np.sum(err ** 2)))

    # per-airport unmatched breakdown, mirroring their census table
    if "ADEP_mvt" in meta.columns:
        airport = meta["ADEP_mvt"].astype(str).to_numpy()
    else:
        airport = X["ADEP_mvt"].astype(str).to_numpy()
    print("\n  per-airport unmatched (n, share of that airport's rows, RMSE, "
          "share of TOTAL unmatched SSE):")
    total_unmatched_sse = np.sum(err[~matched] ** 2)
    for a in sorted(set(airport)):
        m = (~matched) & (airport == a)
        if m.sum() == 0:
            continue
        airport_total = (airport == a).sum()
        r = rmse(err[m])
        sse_share = np.sum(err[m] ** 2) / total_unmatched_sse if total_unmatched_sse > 0 else 0.0
        print("    {:6s} n={:5d}  ({:.2%} of airport rows)  RMSE={:8.2f}  "
              "{:.1%} of total unmatched SSE".format(
                  a, int(m.sum()), m.sum() / airport_total, r, sse_share))


def main() -> int:
    for part in ("val", "test"):
        check(part)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
