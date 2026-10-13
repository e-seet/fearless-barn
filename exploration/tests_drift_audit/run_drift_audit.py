#!/usr/bin/env python3
"""Feature-distribution drift audit: train2025 vs. the real ranking.parquet.

Why: `DATA_QUALITY_AUDIT.md` covered missingness/duplicates/identifier
consistency but not a systematic per-feature distribution comparison. An
outside competitor (`conversation.md` sec.4) reported checking all 52 numeric
features, 2025 Jan+Jul against the 2026 ranking set, "none shifted more than
0.25 sd" -- this repo has no persisted artifact confirming that scope was
ever replicated here (`DATA_QUALITY_AUDIT.md` cites a scratchpad script that
does not survive across sessions), so this is a genuine gap, not a
re-litigation of a closed result.

This is a bug-detector, not a lever: a clean result buys confidence before a
future submission; a dirty result on a feature the model leans on heavily
(delta_sched_sec, the congestion columns) would be the first mechanical
explanation found for the ~47-51s local-to-board RMSE offset every arm in
this family carries (`DATA_QUALITY_AUDIT.md`'s join-failure-rate finding is
a *partial* explanation; this checks for a second, independent one). Either
outcome is diagnostic, not a score improvement.

Two comparisons, both restricted to PHASE_mvt=="DEP" at the 10 target
airports (the scored population):

1. **Full training pool vs. each scored month separately.** What the model
   was actually trained on (all 12 months of 2025) vs. what it scores
   against in the ranking set (Jan 2026, Jul 2026 -- Jul restricted to the
   3 airports the ranking file actually covers, matching
   `config.SPLITS["seasonal"]`). Operationally the most relevant comparison
   -- this is the actual train/serve gap.
2. **Same-calendar-month, year-over-year** (train's own Jan/Jul 2025 vs.
   ranking's Jan/Jul 2026). Isolates genuine year-boundary drift by removing
   the seasonal-composition confound that (1) can't separate from a
   year-boundary effect -- this is the comparison the outside competitor's
   report and the community's methodology in `conversation.md` sec.4 used.

Threshold: |z| > 0.25 (rank_mean - train_mean, standardized by the train
column's own SD), pre-registered to match the community's own reported
figure so results are directly comparable. Pure descriptive statistics on
already-featurised data -- no training, no `features.py` edit.
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
from common import dataset as D
from common import io as cio
from common import features as F

FEATURESET = "with_aobt"
SD_THRESHOLD = 0.25
SEASONAL_JUL_AIRPORTS = config.SPLITS["seasonal"]["airport_filter"]["2025-07"]


def numeric_cols(X: "pd.DataFrame") -> list:
    exclude = set(F.CATEGORICAL) | {config.ID, "ADEP_mvt", "MVT_TIME_UTC_mvt", "month"}
    return [c for c in X.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(X[c])]


def compare(train_df, rank_df, cols, label):
    rows = []
    for c in cols:
        tr = train_df[c].astype("float64")
        rk = rank_df[c].astype("float64")
        tr_mean, tr_std = tr.mean(), tr.std(ddof=0)
        rk_mean = rk.mean()
        if not np.isfinite(tr_std) or tr_std == 0:
            z = float("nan")
        else:
            z = (rk_mean - tr_mean) / tr_std
        rows.append({
            "comparison": label, "feature": c,
            "train_n": int(tr.notna().sum()), "rank_n": int(rk.notna().sum()),
            "train_mean": round(float(tr_mean), 4), "rank_mean": round(float(rk_mean), 4),
            "train_std": round(float(tr_std), 4),
            "z": round(float(z), 4) if np.isfinite(z) else None,
            "flagged": bool(np.isfinite(z) and abs(z) > SD_THRESHOLD),
        })
    return rows


def main() -> int:
    print("[drift audit] loading train2025 features ({})...".format(FEATURESET))
    Xtr = D.build_scope("train2025", FEATURESET)
    tr_meta = cio.load_training(columns=[config.ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    tr_meta = tr_meta[tr_meta.PHASE_mvt.eq("DEP")]
    Xtr = Xtr.merge(tr_meta[[config.ID, "MVT_TIME_UTC_mvt"]], on=config.ID, how="inner")
    Xtr["month_of_year"] = pd.to_datetime(Xtr["MVT_TIME_UTC_mvt"]).dt.strftime("%m")
    print("  train DEP rows: {}".format(len(Xtr)))

    print("[drift audit] loading ranking features ({})...".format(FEATURESET))
    Xrk, rk_meta = D.load_ranking_features(FEATURESET)
    Xrk = Xrk.reset_index()
    rk_meta2 = rk_meta.reset_index()[[config.ID, "MVT_TIME_UTC_mvt"]]
    Xrk = Xrk.merge(rk_meta2, on=config.ID, how="inner")
    Xrk["month_of_year"] = pd.to_datetime(Xrk["MVT_TIME_UTC_mvt"]).dt.strftime("%m")
    print("  ranking DEP rows: {}".format(len(Xrk)))
    print("  ranking month counts:\n{}".format(Xrk["month_of_year"].value_counts().to_string()))
    print("  ranking airports per month:")
    for mo, grp in Xrk.groupby("month_of_year"):
        print("    {}: {}".format(mo, sorted(grp["ADEP_mvt"].unique())))

    cols = numeric_cols(Xrk)
    print("[drift audit] {} numeric feature columns compared".format(len(cols)))

    all_rows = []

    # ---- Comparison 1: full training pool vs. each scored ranking month ----
    for mo in sorted(Xrk["month_of_year"].unique()):
        rk_m = Xrk[Xrk["month_of_year"].eq(mo)]
        airports = sorted(rk_m["ADEP_mvt"].unique())
        tr_pool = Xtr[Xtr["ADEP_mvt"].isin(airports)]  # match ranking's airport composition that month
        label = "full_train_vs_ranking_{}".format(mo)
        all_rows += compare(tr_pool, rk_m, cols, label)

    # ---- Comparison 2: same calendar month, year-over-year ----
    for mo in sorted(Xrk["month_of_year"].unique()):
        rk_m = Xrk[Xrk["month_of_year"].eq(mo)]
        airports = sorted(rk_m["ADEP_mvt"].unique())
        tr_m = Xtr[Xtr["month_of_year"].eq(mo) & Xtr["ADEP_mvt"].isin(airports)]
        label = "same_month_yoy_{}".format(mo)
        all_rows += compare(tr_m, rk_m, cols, label)

    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / "drift_summary.csv", index=False)

    lines = ["feature-distribution drift audit  (train2025 vs. ranking.parquet, |z| > {} sd flagged)".format(SD_THRESHOLD),
             "=" * 90]
    for label in df["comparison"].unique():
        sub = df[df["comparison"].eq(label)]
        flagged = sub[sub["flagged"]]
        lines.append("\n-- {} -- {} of {} features flagged".format(label, len(flagged), len(sub)))
        if len(flagged):
            for _, r in flagged.sort_values("z", key=lambda s: s.abs(), ascending=False).iterrows():
                lines.append("   {:32s} z={:+7.3f}  train_mean={:12.3f}  rank_mean={:12.3f}  train_std={:10.3f}  (n_train={}, n_rank={})".format(
                    r["feature"], r["z"], r["train_mean"], r["rank_mean"], r["train_std"], r["train_n"], r["rank_n"]))
        else:
            lines.append("   (none)")
    txt = "\n".join(lines)
    (HERE / "drift_summary.txt").write_text(txt + "\n", encoding="utf-8")
    print("\n" + txt)
    print("\n[drift audit] wrote {} and drift_summary.txt".format(HERE / "drift_summary.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
