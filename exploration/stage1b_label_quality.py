#!/usr/bin/env python3
"""Stage 1b -- label quality and temporal drift.

Stage 1 checks that the data is STRUCTURALLY sound. This stage asks whether the
label is TRUSTWORTHY and whether 2025 still describes 2026, using the one lever
available: the dataset ships two independent off-block timestamps.

  * BLOCK_TIME_UTC_mvt -- the movement system's off-block. The target is defined
    from it (MVT_TIME - BLOCK_TIME), and it is blanked in the scored set.
  * AOBT_3_flt -- the flown-trajectory off-block, from the Network Manager side.
    Present in the scored set.

Where the two disagree, at least one is wrong. Disagreement is therefore a
per-airport label-noise floor: an irreducible error no model can predict away.

Output: artifacts/audit/label_quality.json
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[1]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from common import io as cio
from common.bootstrap import write_json_guarded

COLS = ["PHASE_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    config.AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    rep = {"stage": "stage1b_label_quality"}

    tr = cio.load_training(columns=COLS)
    tr = tr[tr.PHASE_mvt.eq("DEP")].copy()
    tr["y"] = tr[config.TARGET].astype("float64")
    tr["proxy"] = (tr.MVT_TIME_UTC_mvt - tr.AOBT_3_flt).dt.total_seconds()
    tr["absdiff"] = (tr.proxy - tr.y).abs()
    tr["month"] = tr.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")

    rk = cio.load_ranking(columns=COLS)
    rk = rk[rk.PHASE_mvt.eq("DEP")].copy()
    rk["proxy"] = (rk.MVT_TIME_UTC_mvt - rk.AOBT_3_flt).dt.total_seconds()
    rk["month"] = rk.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")

    # ---------------------------------------------------- source disagreement
    print("\n== label-noise floor: do the two off-block sources agree? ==")
    g = tr.dropna(subset=["proxy"]).groupby("ADEP_mvt", observed=True)
    dis = pd.DataFrame({
        "n": g.size(),
        "agree_within_60s_pct": g.apply(lambda x: float((x.absdiff <= 60).mean() * 100),
                                        include_groups=False),
        "agree_within_300s_pct": g.apply(lambda x: float((x.absdiff <= 300).mean() * 100),
                                         include_groups=False),
        "median_absdiff_s": g.absdiff.median(),
        # RMS of the source disagreement -- a floor on achievable RMSE if the
        # label is the noisy one of the pair
        "rms_disagreement_s": g.apply(
            lambda x: float(np.sqrt(np.mean((x.proxy - x.y) ** 2))), include_groups=False),
        "label_p99_s": g.y.quantile(0.99),
        "proxy_p99_s": g.proxy.quantile(0.99),
        "label_frac_gt_7200_pct": g.apply(lambda x: float((x.y > 7200).mean() * 100),
                                          include_groups=False),
        "proxy_frac_gt_7200_pct": g.apply(lambda x: float((x.proxy > 7200).mean() * 100),
                                          include_groups=False),
    })
    dis["label_vs_proxy_p99_ratio"] = dis.label_p99_s / dis.proxy_p99_s
    print(dis.round(2).sort_values("agree_within_300s_pct").to_string())
    rep["source_disagreement"] = dis.round(4).to_dict(orient="index")

    # ------------------------------------------------------------- the tail
    print("\n== target tail (train DEP label) vs proxy tail (train and scored) ==")
    tails = {}
    for name, s in (("train_label", tr.y.dropna()),
                    ("train_proxy", tr.proxy.dropna()),
                    ("rank_proxy", rk.proxy.dropna())):
        tails[name] = {"n": int(s.size), "p50": float(s.median()),
                       "p99": float(s.quantile(.99)), "p999": float(s.quantile(.999)),
                       "max": float(s.max()),
                       "frac_gt_7200_pct": float((s > 7200).mean() * 100),
                       "frac_lt_60_pct": float((s < 60).mean() * 100)}
        print("  {:12s} n={:8d} p50={:6.0f} p99={:7.0f} p99.9={:8.0f} max={:8.0f} "
              ">7200s={:.4f}% <60s={:.4f}%".format(
                  name, tails[name]["n"], tails[name]["p50"], tails[name]["p99"],
                  tails[name]["p999"], tails[name]["max"],
                  tails[name]["frac_gt_7200_pct"], tails[name]["frac_lt_60_pct"]))
    rep["tails"] = tails
    print("  NOTE: the scored set's LABEL tail is unobservable (blanked). The train")
    print("        proxy is far cleaner than the train label, so a clean rank proxy")
    print("        is NOT evidence of a clean rank label -- do not assume one.")
    rep["tail_caveat"] = (
        "rank label tail unobservable; train proxy is already much cleaner than the "
        "train label (0.0009% vs 0.0280% >7200s), so rank proxy cleanliness carries "
        "no information about the rank label tail")

    # ------------------------------------------------------------- drift
    print("\n== temporal drift 2025 -> 2026, measured on the proxy (comparable both sides) ==")
    a = tr.groupby("ADEP_mvt", observed=True).agg(
        label_median_2025=("y", "median"), proxy_median_2025=("proxy", "median"))
    b = rk.groupby("ADEP_mvt", observed=True).proxy.median().rename("proxy_median_2026")
    dr = a.join(b)
    dr["drift_s"] = dr.proxy_median_2026 - dr.proxy_median_2025
    dr["drift_pct"] = dr.drift_s / dr.proxy_median_2025 * 100
    w = rk.ADEP_mvt.value_counts(normalize=True)
    dr["scored_share_pct"] = (w * 100).reindex(dr.index)
    print(dr.round(2).sort_values("drift_pct", ascending=False).to_string())
    rep["drift"] = dr.round(4).to_dict(orient="index")

    dr2 = dr.dropna(subset=["drift_s", "scored_share_pct"])
    rep["scored_weighted_abs_drift_s"] = float(
        (dr2.drift_s.abs() * dr2.scored_share_pct / 100).sum())
    print("\n  scored-share-weighted mean |drift|: {:.1f} s".format(
        rep["scored_weighted_abs_drift_s"]))

    # per-month proxy median, so a seasonal effect is separable from a level shift
    rep["rank_proxy_median_by_airport_month"] = {
        "{}|{}".format(a_, m): float(v) for (a_, m), v in
        rk.dropna(subset=["proxy"]).groupby(["ADEP_mvt", "month"], observed=True)
        .proxy.median().items()}
    rep["train_proxy_median_by_airport_month"] = {
        "{}|{}".format(a_, m): float(v) for (a_, m), v in
        tr.dropna(subset=["proxy"]).groupby(["ADEP_mvt", "month"], observed=True)
        .proxy.median().items()}

    write_json_guarded(config.AUDIT_DIR / "label_quality.json", rep, force=args.force)
    print("\n  report: {}".format(config.AUDIT_DIR / "label_quality.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
