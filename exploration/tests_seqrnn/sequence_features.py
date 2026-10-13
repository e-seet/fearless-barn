#!/usr/bin/env python3
"""Per-departure BACKWARD-ONLY movement sequence tensors, for a recurrent
(GRU) state encoder -- the genuinely sequential construction, as opposed to
`pipeline_queueattn_antigravity`'s fixed-K bidirectional attention pool.

Two differences from queue-attention, both deliberate:

1. **Causal, not bidirectional.** Queue-attention picks the K temporally-
   NEAREST other departures on EITHER side of the query (future movements
   included, legitimate because ranking.parquet ships every movement in the
   scored months). This instead takes only the K movements strictly BEFORE
   the query, in true chronological order, so a GRU can process them as an
   actual sequence and produce a single "traffic state right now" summary --
   the shape an airport's controllers/pilots would actually experience,
   and a materially different inductive bias than a symmetric neighbour set.

2. **Both phases, not DEP-only.** `neighbors.py` restricts its candidate
   pool to other DEPARTURES only (indexes into `dep_idx` before building the
   per-airport candidate arrays). Arrivals also drive ground congestion
   (runway crossings, apron occupancy) -- already true in `build_congestion`'s
   `arr_prev_*m` scalars -- so this sequence includes both DEP and ARR
   movements at the airport, each tagged with an `is_dep` flag.

Ranking-set safety identical to `neighbors.py`: only `TAXITIME_SEC_mvt` and
`BLOCK_TIME_UTC_mvt` are blanked on DEP rows; a neighbour's raw taxi time is
never used. `proxy_taxi_sec` (`MVT_TIME_UTC_mvt - AOBT_3_flt`) is the same
already-safe proxy, valid only for DEP neighbours with an AOBT; ARR
neighbours contribute their existence/timing/runway but no taxi-time proxy
(masked to 0, `is_dep=0` tells the model why).

No time-window cap: the K most recent movements before the query, however
far back that reaches. A queue-attention-style 90-minute cap would silently
degenerate to "mostly masked" during this dataset's quiet overnight/off-peak
hours -- the sequence's whole point is to hand the GRU whatever state is
actually available, which the model is free to learn contributes ~nothing
when the gap is large. `delta_t_min` (time before the query) tells it the
gap directly.
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

K = 24
SOURCE_COLUMNS = ("MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt",
                  "MVT_TIME_UTC_mvt", "AOBT_3_flt", "RUNWAY_mvt")
# per-step features: delta_t_min, is_dep, same_runway, proxy_taxi_sec, mask
SEQ_FEATURE_DIM = 4   # mask is returned separately, used both as a feature and a loss gate


def code_fingerprint() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def build_sequence_tensors(mov: "pd.DataFrame", k: int = K) -> dict:
    """Returns arrays aligned to `mov`'s DEP rows (DEP-row order, not sorted).

    Keys: 'mvt_id' (n,); 'seq' (n, k, SEQ_FEATURE_DIM) float32, OLDEST first
    (index 0) to MOST RECENT last (index k-1), so a GRU fed in order 0..k-1
    naturally ends at "now"; 'mask' (n, k) float32 (1 = real event).
    A masked slot is all-zero in 'seq' too.
    """
    missing = set(SOURCE_COLUMNS) - set(mov.columns)
    if missing:
        raise KeyError("build_sequence_tensors needs {}".format(sorted(missing)))

    is_dep_all = mov["PHASE_mvt"].eq("DEP").to_numpy()
    dep_idx = np.flatnonzero(is_dep_all)
    n_total = dep_idx.size

    # candidate pool = EVERY movement (DEP+ARR) at the airport, not DEP-only
    airport_all = mov["ADEP_mvt"].to_numpy()
    airport_all = np.where(is_dep_all, mov["ADEP_mvt"].to_numpy(), mov["ADES_mvt"].to_numpy())
    mvt_ns_all = _to_ns(mov["MVT_TIME_UTC_mvt"]).astype("float64")
    aobt_ns_all = _to_ns(mov["AOBT_3_flt"]).astype("float64")
    aobt_valid_all = mov["AOBT_3_flt"].notna().to_numpy()
    rwy_all = mov["RUNWAY_mvt"].astype("object").to_numpy()
    proxy_all = (mvt_ns_all - aobt_ns_all) / 1e9
    proxy_all[(proxy_all <= 0) | (proxy_all > 86400.0) | ~aobt_valid_all] = np.nan

    query_mvt_id = mov["MVT_ID_mvt"].to_numpy()[dep_idx]
    query_airport = airport_all[dep_idx]
    query_mvt_ns = mvt_ns_all[dep_idx]
    query_rwy = rwy_all[dep_idx]

    out_seq = np.zeros((n_total, k, SEQ_FEATURE_DIM), dtype="float32")
    out_mask = np.zeros((n_total, k), dtype="float32")

    for ap in pd.unique(query_airport):
        pool_sel = np.flatnonzero(airport_all == ap)
        if pool_sel.size == 0:
            continue
        order = np.argsort(mvt_ns_all[pool_sel], kind="stable")
        pool_sorted = pool_sel[order]                 # global indices, time-ascending
        pool_t = mvt_ns_all[pool_sorted]
        pool_is_dep = is_dep_all[pool_sorted]
        pool_rwy = rwy_all[pool_sorted]
        pool_proxy = proxy_all[pool_sorted]

        query_sel = np.flatnonzero(query_airport == ap)   # positions into dep_idx/query arrays
        # position of each query's global index within pool_sorted (queries ARE in the pool)
        pos_in_pool = np.searchsorted(pool_t, query_mvt_ns[query_sel], side="left")
        # searchsorted can land on ties/self; step back past any entries at/after the query time
        for qi, p in zip(query_sel, pos_in_pool):
            p = int(p)
            # walk back past ties at the exact same timestamp (avoid seeing "self" or
            # simultaneous-but-not-yet-happened events as if they were in the past)
            while p > 0 and pool_t[p - 1] >= query_mvt_ns[qi]:
                p -= 1
            lo = max(0, p - k)
            n_avail = p - lo
            if n_avail <= 0:
                continue
            src = pool_sorted[lo:p]                   # oldest..newest, length n_avail
            slot_start = k - n_avail                   # right-align: most recent at index k-1
            dt_min = (query_mvt_ns[qi] - mvt_ns_all[src]) / 1e9 / 60.0
            is_dep_step = is_dep_all[src].astype("float32")
            same_rwy = (rwy_all[src] == query_rwy[qi]).astype("float32")
            proxy_step = proxy_all[src]
            proxy_ok = np.isfinite(proxy_step)
            out_seq[qi, slot_start:k, 0] = dt_min.astype("float32")
            out_seq[qi, slot_start:k, 1] = is_dep_step
            out_seq[qi, slot_start:k, 2] = same_rwy
            out_seq[qi, slot_start:k, 3] = np.where(proxy_ok, proxy_step, 0.0).astype("float32")
            out_mask[qi, slot_start:k] = 1.0

    return {"mvt_id": query_mvt_id, "seq": out_seq, "mask": out_mask}


def _cache_path(scope: str) -> Path:
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    return config.FEATURE_DIR / "seqrnn_neighbors__{}__{}.npz".format(
        scope, code_fingerprint())


def load_scope_cached(scope: str, *, force: bool = False) -> dict:
    from common import io as cio

    path = _cache_path(scope)
    if path.exists() and not force:
        z = np.load(path)
        return {k: z[k] for k in z.files}

    stale = sorted(config.FEATURE_DIR.glob("seqrnn_neighbors__{}__*.npz".format(scope)))
    if stale:
        print("[seqrnn] sequence_features.py changed -- rebuilding {} (superseding {})".format(
            scope, [p.name for p in stale]))

    if scope == "train2025":
        mov = cio.load_training(columns=list(SOURCE_COLUMNS))
    elif scope == "ranking":
        mov = cio.load_ranking(columns=list(SOURCE_COLUMNS))
    else:
        raise ValueError("unknown scope {!r}".format(scope))

    out = build_sequence_tensors(mov)
    np.savez(path, **out)
    return out


def align_to_ids(tensors: dict, ids: "np.ndarray") -> dict:
    idx = pd.Index(tensors["mvt_id"])
    pos = idx.get_indexer(np.asarray(ids))
    n, k, f = tensors["seq"].shape
    found = pos >= 0
    seq = np.zeros((len(pos), k, f), dtype="float32")
    mask = np.zeros((len(pos), k), dtype="float32")
    seq[found] = tensors["seq"][pos[found]]
    mask[found] = tensors["mask"][pos[found]]
    if not np.all(found):
        print("[seqrnn] {} of {} ids not found in sequence cache -- "
              "using all-zero/fully-masked rows for them".format(
                  int((~found).sum()), len(pos)))
    return {"seq": seq, "mask": mask}
