"""Data loading with the ranking-availability contract enforced at load time.

The single most dangerous thing this build can do is train on a column that
ranking.parquet blanks. `load_movements(apply_ranking_contract=True)` makes that
structurally impossible for the DEP rows a model trains on: the blanked columns
are dropped from the feature frame before any pipeline sees them, and the target
is carried in a separate Series.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


def training_files() -> "list[Path]":
    files = sorted(config.DATA_DIR.glob("training_*.parquet"))
    if not files:
        raise FileNotFoundError(
            "no training_*.parquet under {}. Download the challenge data first "
            "(see submit.py --help for the mc commands).".format(config.DATA_DIR)
        )
    return files


def _month(series: "pd.Series") -> "pd.Series":
    """'YYYY-MM' of a tz-aware timestamp column, without the tz-drop warning."""
    return series.dt.strftime("%Y-%m")


def load_raw(paths, columns=None) -> "pd.DataFrame":
    frames = [pq.read_table(p, columns=columns).to_pandas() for p in paths]
    df = pd.concat(frames, ignore_index=True)
    return df


def _with_month(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add the 'YYYY-MM' partition key, but only when the timestamp was loaded.

    A caller that passes an explicit `columns=` list to save memory must not be
    forced to include MVT_TIME_UTC_mvt just so this helper can run.
    """
    if "MVT_TIME_UTC_mvt" in df.columns:
        df["month"] = _month(df["MVT_TIME_UTC_mvt"])
    return df


def load_training(columns=None) -> "pd.DataFrame":
    return _with_month(load_raw(training_files(), columns=columns))


def load_ranking(columns=None) -> "pd.DataFrame":
    return _with_month(load_raw([config.RANKING_PARQUET], columns=columns))


def apply_ranking_contract(df: "pd.DataFrame") -> "pd.DataFrame":
    """Blank on DEP rows exactly what ranking.parquet blanks.

    Applied to TRAINING data so a feature builder cannot accidentally read a
    column that will be NaN at scoring time. The target is extracted separately
    by `split_target` BEFORE this runs.
    """
    out = df.copy()
    dep = out["PHASE_mvt"].eq("DEP")
    for col in config.BLANKED_ON_RANKING_DEP:
        if col in out.columns:
            out.loc[dep, col] = pd.NaT if pd.api.types.is_datetime64_any_dtype(out[col]) else np.nan
    return out


def split_target(df: "pd.DataFrame"):
    """Return (frame_without_target_on_dep, dep_target_series_indexed_like_df)."""
    y = df[config.TARGET].astype("float64").where(df["PHASE_mvt"].eq("DEP"))
    return df, y


def assert_ranking_computable(feature_cols, *, name="features") -> dict:
    """Hard gate: every feature source column must survive into ranking DEP rows.

    Raises rather than warns. A feature that reads a blanked column produces a
    model that scores NaN on the leaderboard -- catching it here costs a second,
    catching it after a GPU run costs the run.
    """
    banned = set(config.BLANKED_ON_RANKING_DEP)
    offending = sorted(c for c in feature_cols if c in banned)
    if offending:
        raise AssertionError(
            "{} reads column(s) that ranking.parquet blanks on DEP rows: {}. "
            "These are unavailable at scoring time -- the model would receive NaN "
            "for every scored row.".format(name, offending)
        )
    return {"check": "ranking_computable", "name": name,
            "n_cols": len(list(feature_cols)), "banned": sorted(banned), "passed": True}
