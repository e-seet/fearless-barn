#!/usr/bin/env python3
"""Realised-taxi congestion: the mean PROXY taxi-out of departures that have
already left, at this airport, in the last 30 / 60 minutes.

Why this is not already in features.py
--------------------------------------
`common/features.py` has ten traffic-density columns, but its rolling-mean
helper is used exactly once -- for `arr_taxiin_mean_prev_60m`, arrivals only.
That was deliberate and correct: departures' realised taxi-out is
`TAXITIME_SEC_mvt`, and both it and `BLOCK_TIME_UTC_mvt` are blanked on DEP rows
in `ranking.parquet` (`config.BLANKED_ON_RANKING_DEP`), so a feature built from
them could not be computed for the scored set at all.

The PROXY can. `MVT_TIME_UTC_mvt` (takeoff) and `AOBT_3_flt` (off-block) are both
in `config.DEP_AVAILABLE`, so `takeoff - AOBT_3` is available for every scored
DEP row. Averaged over departures that ALREADY took off, it is the most direct
congestion signal the data admits -- it measures how long taxiing is actually
taking right now, rather than inferring it from movement counts.

The counts already present (`dep_prev_30m`, `same_rwy_prev_30m`, ...) say how
BUSY it is. This says how SLOW it is. Those are different: a busy airport running
smoothly and a quiet one with a blocked taxiway have similar counts and very
different taxi times.

Why the proxy is safe here
--------------------------
* Only `DEP_AVAILABLE` columns are read; `assert_ranking_computable` enforces it.
* `TAXITIME_SEC_mvt` is never touched -- this is a timestamp difference.
* Strictly backward-looking: `[t - window, t)`, so a row cannot see itself, and
  ties at exactly `t` are excluded by the left-side searchsorted.
* Present in BOTH featuresets (`no_aobt` falls back to `EOBT_1`), so the
  "what is AOBT worth" ablation does not silently also measure this -- the same
  reasoning as `n_taxiing_at_pushback` (design-gate S3).

Cached separately and keyed on THIS file's hash, like
`pipeline_lightgbm_arrfeat/extra.py`: adding these columns to `features.py`
instead would change `config.feature_code_fingerprint()` and make every trained
run and every built submission in the repo stale.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import io as cio
from common.features import _rolling_prev_mean, _to_ns

WINDOWS_MIN = (30, 60)
ADDED_COLUMNS = tuple(
    ["dep_realised_taxi_mean_prev_{}m".format(w) for w in WINDOWS_MIN]
    + ["dep_realised_taxi_mean_rwy_prev_{}m".format(w) for w in WINDOWS_MIN]
    + ["dep_realised_taxi_n_prev_60m"]
)

SOURCE_COLUMNS = ("MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
                  "MVT_TIME_UTC_mvt", "RUNWAY_mvt", "AOBT_3_flt", "EOBT_1_flt")


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def _proxy_taxi(mov: "pd.DataFrame", featureset: str) -> "np.ndarray":
    """takeoff - off-block, in seconds, by the best estimate this featureset allows.

    NaN (not 0) where no off-block estimate exists: 0 would assert "instant
    pushback", a different claim from "unknown", and would drag every window mean
    it entered towards zero.
    """
    src = "AOBT_3_flt" if featureset == "with_aobt" else "EOBT_1_flt"
    t = _to_ns(mov["MVT_TIME_UTC_mvt"]).astype("float64")
    ob = _to_ns(mov[src]).astype("float64")
    v = (t - ob) / 1e9
    # A negative or absurd proxy is a bad timestamp pairing, not a long taxi.
    v[(v <= 0) | (v > 86400.0)] = np.nan
    return v


def build_extra(mov: "pd.DataFrame", featureset: str) -> "pd.DataFrame":
    missing = set(SOURCE_COLUMNS) - set(mov.columns)
    if missing:
        raise KeyError("build_extra needs {}".format(sorted(missing)))

    is_dep = mov["PHASE_mvt"].eq("DEP").to_numpy()
    airport = np.where(is_dep, mov["ADEP_mvt"].to_numpy(), mov["ADES_mvt"].to_numpy())
    t_ns = _to_ns(mov["MVT_TIME_UTC_mvt"])
    proxy = _proxy_taxi(mov, featureset)
    rwy = mov["RUNWAY_mvt"].astype("object").to_numpy()

    dep_idx = np.flatnonzero(is_dep)
    n = dep_idx.size
    pos_of_row = np.full(len(mov), -1, dtype="int64")
    pos_of_row[dep_idx] = np.arange(n)

    cols = {c: np.full(n, np.nan, dtype="float64") for c in ADDED_COLUMNS}
    minute = 60 * 1_000_000_000

    for a in pd.unique(airport):
        m_ap = (airport == a) & is_dep
        rows = np.flatnonzero(m_ap)
        if rows.size == 0:
            continue
        order = rows[np.argsort(t_ns[rows], kind="stable")]
        ev_t, ev_v = t_ns[order], proxy[order]
        pos = pos_of_row[order]

        for w in WINDOWS_MIN:
            cols["dep_realised_taxi_mean_prev_{}m".format(w)][pos] = \
                _rolling_prev_mean(ev_t, ev_v, ev_t, w * minute)

        # how many of those observations the mean rests on -- a mean over two
        # flights and a mean over forty are not the same evidence, and the model
        # can only discount the thin one if it can see the count
        ok = np.isfinite(ev_v)
        ccnt = np.concatenate([[0], np.cumsum(ok.astype("int64"))])
        left = np.searchsorted(ev_t, ev_t - 60 * minute, side="left")
        right = np.searchsorted(ev_t, ev_t, side="left")
        cols["dep_realised_taxi_n_prev_60m"][pos] = (ccnt[right] - ccnt[left]).astype("float64")

        for r in pd.unique(rwy[order]):
            sel = rwy[order] == r
            if sel.sum() == 0:
                continue
            rt, rv, rp = ev_t[sel], ev_v[sel], pos[sel]
            for w in WINDOWS_MIN:
                cols["dep_realised_taxi_mean_rwy_prev_{}m".format(w)][rp] = \
                    _rolling_prev_mean(rt, rv, rt, w * minute)

    return pd.DataFrame(cols, index=mov.index[dep_idx])


def build_scope_extra(scope: str, featureset: str, *, force: bool = False):
    """Whole-scope build, cached and keyed on this file's hash (S6 discipline)."""
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FEATURE_DIR / "congest__{}__{}__{}.parquet".format(
        scope, featureset, code_fingerprint())
    if path.exists() and not force:
        return pd.read_parquet(path)
    cio.assert_ranking_computable(list(SOURCE_COLUMNS),
                                  name="per_airport_congest/{}".format(featureset))
    mov = (cio.load_training(columns=list(SOURCE_COLUMNS)) if scope == "train2025"
           else cio.load_ranking(columns=list(SOURCE_COLUMNS)))
    X = build_extra(mov, featureset)
    X[config.ID] = mov.loc[X.index, config.ID].to_numpy()
    X = X.set_index(config.ID)
    X.to_parquet(path)
    print("  [congest] built {} columns for {} -> {}".format(
        len(ADDED_COLUMNS), scope, path.name))
    return X


def attach(X, scope, featureset):
    """Join the extra columns on, failing loudly on a gap.

    A silent all-NaN join would read as 'the new features did not help', which is
    the most expensive way for this experiment to be wrong.
    """
    ex = build_scope_extra(scope, featureset)
    add = ex.reindex(X.index)
    miss = int(add.isna().all(axis=1).sum())
    if miss:
        raise RuntimeError(
            "{} of {} rows have no congest row -- the cache does not cover this "
            "scope.".format(miss, len(X)))
    return pd.concat([X, add], axis=1)
