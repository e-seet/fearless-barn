#!/usr/bin/env python3
"""Per-flight departure-queue neighbour tensors, for attention pooling.

Upgrades the existing scalar congestion aggregates (`dep_prev_30m`,
`same_rwy_prev_15m`, `dep_realised_taxi_mean_prev_60m`, ...) into an actual
SET of the K temporally-nearest other departures at the same airport, so a
model can attend over individual neighbours instead of only ever seeing
pre-collapsed counts/means. The scalar aggregates are lossy (a count of 3
throws away which 3, their runway, and how fast they actually moved); this
does not collapse that information before the model sees it.

Ranking-set safety, same rule as `common/features.py` and
`pipeline_per_airport_congest/extra.py`: only `TAXITIME_SEC_mvt` and
`BLOCK_TIME_UTC_mvt` are blanked on DEP rows in ranking.parquet (verified
directly: every other DEP-available column, including MVT_TIME_UTC_mvt and
AOBT_3_flt, is 100%/98.5% populated there). So a neighbour's raw
TAXITIME_SEC_mvt can NEVER be used (blanked for every DEP row, not just the
query), but the proxy `MVT_TIME_UTC_mvt - AOBT_3_flt` is safe for every
neighbour exactly as it already is for the scalar congestion features.
Both directions of the time window (before AND after the query's own
off-block) are legitimate for the same reason `build_congestion`'s
`_next` windows are: ranking.parquet ships every movement in the scored
months, so "what else moved around this flight" is observable at scoring
time regardless of order.

Must be computed over the WHOLE period before subsetting to a split part
(same rule as `build_scope`'s docstring) -- computing per-part would let a
val row see a truncated neighbourhood a train row saw in full, a silent
train/serve skew.

K-nearest-by-time-distance in a SORTED array is always a subset of the
K positions immediately adjacent in index on each side: sortedness means
index distance IS rank distance, so a candidate outside index window
[i-K, i+K] can never be closer in value than every one of the K points
between it and i. Gathering exactly those 2K candidates and picking the
K smallest |delta t| among them is therefore exact, not an approximation.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common.features import _to_ns

K = 8
WINDOW_MIN = 90.0
SOURCE_COLUMNS = ("MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
                  "AOBT_3_flt", "RUNWAY_mvt")
NEIGHBOR_FEATURES = ("delta_t_min", "same_runway", "proxy_taxi_sec")


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def build_neighbor_tensors(mov: "pd.DataFrame", k: int = K,
                           window_min: float = WINDOW_MIN) -> dict:
    """Returns arrays aligned to `mov`'s DEP rows, in DEP-row order (NOT sorted).

    Keys: 'mvt_id' (n,), 'delta_t_min'/'same_runway'/'proxy_taxi_sec'/'mask'
    all (n, k). A masked slot (mask=0) has all three features set to 0 --
    the model must learn from the mask, not from a 0 meaning "simultaneous
    same-runway zero-taxi neighbour".
    """
    missing = set(SOURCE_COLUMNS) - set(mov.columns)
    if missing:
        raise KeyError("build_neighbor_tensors needs {}".format(sorted(missing)))

    is_dep = mov["PHASE_mvt"].eq("DEP").to_numpy()
    dep_idx = np.flatnonzero(is_dep)
    n_total = dep_idx.size

    airport = mov["ADEP_mvt"].to_numpy()[dep_idx]
    mvt_id = mov["MVT_ID_mvt"].to_numpy()[dep_idx]
    aobt_valid = mov["AOBT_3_flt"].notna().to_numpy()[dep_idx]
    aobt_ns = _to_ns(mov["AOBT_3_flt"])[dep_idx].astype("float64")
    mvt_ns = _to_ns(mov["MVT_TIME_UTC_mvt"])[dep_idx].astype("float64")
    rwy = mov["RUNWAY_mvt"].astype("object").to_numpy()[dep_idx]
    proxy_all = (mvt_ns - aobt_ns) / 1e9
    proxy_all[(proxy_all <= 0) | (proxy_all > 86400.0) | ~aobt_valid] = np.nan

    out_delta = np.zeros((n_total, k), dtype="float32")
    out_samerwy = np.zeros((n_total, k), dtype="float32")
    out_proxy = np.zeros((n_total, k), dtype="float32")
    out_mask = np.zeros((n_total, k), dtype="float32")

    offsets = [o for o in range(-k, k + 1) if o != 0]

    for ap in pd.unique(airport):
        sel = np.flatnonzero((airport == ap) & aobt_valid)
        m = sel.size
        if m == 0:
            continue
        order = np.argsort(aobt_ns[sel], kind="stable")
        sidx = sel[order]                       # dep-local indices, sorted by AOBT3
        t_sorted = aobt_ns[sidx]
        rwy_sorted = rwy[sidx]
        proxy_sorted = proxy_all[sidx]

        pos = np.arange(m)
        cand_delta_min = np.full((m, len(offsets)), np.inf, dtype="float64")
        cand_local = np.full((m, len(offsets)), -1, dtype="int64")
        for ci, o in enumerate(offsets):
            src = pos + o
            ok = (src >= 0) & (src < m)
            cand_local[ok, ci] = src[ok]
            dt_min = (t_sorted[pos[ok]] - t_sorted[src[ok]]) / 1e9 / 60.0
            cand_delta_min[ok, ci] = dt_min

        abs_delta = np.abs(cand_delta_min)
        abs_delta[np.abs(cand_delta_min) > window_min] = np.inf
        pick = np.argsort(abs_delta, axis=1, kind="stable")[:, :k]
        rows = np.arange(m)[:, None]

        chosen_local = cand_local[rows, pick]
        chosen_delta = cand_delta_min[rows, pick]
        chosen_valid = np.isfinite(abs_delta[rows, pick]) & (chosen_local >= 0)

        chosen_local_safe = np.where(chosen_valid, chosen_local, 0)
        chosen_rwy = rwy_sorted[chosen_local_safe]
        chosen_samerwy = (chosen_rwy == rwy_sorted[pos][:, None]).astype("float32")
        chosen_proxy = proxy_sorted[chosen_local_safe]
        proxy_ok = chosen_valid & np.isfinite(chosen_proxy)

        delta_out = np.where(chosen_valid, chosen_delta, 0.0).astype("float32")
        samerwy_out = np.where(chosen_valid, chosen_samerwy, 0.0).astype("float32")
        proxy_out = np.where(proxy_ok, chosen_proxy, 0.0).astype("float32")

        dep_local_rows = sidx  # dep-local row index (into 0..n_total-1) for each sorted position
        out_delta[dep_local_rows] = delta_out
        out_samerwy[dep_local_rows] = samerwy_out
        out_proxy[dep_local_rows] = proxy_out
        out_mask[dep_local_rows] = chosen_valid.astype("float32")

    return {
        "mvt_id": mvt_id,
        "delta_t_min": out_delta,
        "same_runway": out_samerwy,
        "proxy_taxi_sec": out_proxy,
        "mask": out_mask,
    }


def _cache_path(scope: str) -> Path:
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    return config.FEATURE_DIR / "queueattn_neighbors__{}__{}.npz".format(
        scope, code_fingerprint())


def load_scope_cached(scope: str, *, force: bool = False) -> dict:
    """Neighbour tensors for a whole scope ('train2025'), cached to .npz.

    Mirrors `common/dataset.py:build_scope`'s cache-keyed-on-code-fingerprint
    pattern (a stale cache from a since-edited neighbors.py would otherwise
    sit there silently invalidating every downstream run).
    """
    from common import io as cio

    path = _cache_path(scope)
    if path.exists() and not force:
        z = np.load(path)
        return {k: z[k] for k in z.files}

    stale = sorted(config.FEATURE_DIR.glob("queueattn_neighbors__{}__*.npz".format(scope)))
    if stale:
        print("[queueattn] neighbors.py changed -- rebuilding {} (superseding {})".format(
            scope, [p.name for p in stale]))

    if scope == "train2025":
        mov = cio.load_training(columns=list(SOURCE_COLUMNS))
    elif scope == "ranking":
        mov = cio.load_ranking(columns=list(SOURCE_COLUMNS))
    else:
        raise ValueError("unknown scope {!r}".format(scope))

    out = build_neighbor_tensors(mov)
    np.savez(path, **out)
    return out


def align_to_ids(tensors: dict, ids: "np.ndarray") -> dict:
    """Subset/reorder a scope's tensors to match `ids` (e.g. Xtr.index).

    IDs not present in the tensor scope (should not happen for train2025-
    sourced ids) get an all-zero, fully-masked row rather than raising --
    same "unknown means mask 0, not a fabricated value" rule as the builder.
    """
    idx = pd.Index(tensors["mvt_id"])
    pos = idx.get_indexer(np.asarray(ids))
    n, k = tensors["delta_t_min"].shape
    out = {}
    for key in ("delta_t_min", "same_runway", "proxy_taxi_sec", "mask"):
        src = tensors[key]
        gathered = np.zeros((len(pos), k), dtype="float32")
        found = pos >= 0
        gathered[found] = src[pos[found]]
        out[key] = gathered
    if not np.all(pos >= 0):
        n_missing = int((~found).sum())
        print("[queueattn] {} of {} ids not found in neighbor cache -- "
              "using all-zero/fully-masked rows for them".format(n_missing, len(pos)))
    return out
