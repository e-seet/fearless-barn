"""Assemble (X, y, meta) for a split part, with an on-disk feature cache.

Every arm reads its training pool through `load_part`, so the
`assert_training_pool_equality` gate in compare.py is comparing pools that were
built by one code path rather than three lookalikes.

Congestion features are computed over the WHOLE period at once (all airports,
both phases) and only then subset to the requested split part. Computing them
per-part would let a val row see a truncated neighbourhood that a train row saw
in full -- a silent train/serve skew rather than a leak, but just as damaging.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import features as F
from common import io as cio

_MEM: dict = {}


def _cache_path(scope: str, featureset: str) -> Path:
    """Cache path keyed on a fingerprint of the code that produced it (S6).

    Without the fingerprint an edit to features.py leaves a stale cache in place
    and every downstream stage runs happily on the OLD features -- succeeding,
    reporting nothing, and invalidating the result.
    """
    config.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    return config.FEATURE_DIR / "{}__{}__{}.parquet".format(
        scope, featureset, config.feature_code_fingerprint())


def build_scope(scope: str, featureset: str, *, force: bool = False) -> "pd.DataFrame":
    """Featurise an entire scope ('train2025' or 'ranking'), cached to parquet."""
    key = (scope, featureset)
    if key in _MEM:
        return _MEM[key]
    path = _cache_path(scope, featureset)
    if path.exists() and not force:
        X = pd.read_parquet(path)
        _MEM[key] = X
        return X
    stale = sorted(config.FEATURE_DIR.glob("{}__{}__*.parquet".format(scope, featureset)))
    if stale:
        print("[dataset] features.py changed -- rebuilding {}/{} (superseding {})".format(
            scope, featureset, [q.name for q in stale]))

    if scope == "train2025":
        mov = cio.load_training()
    elif scope == "ranking":
        mov = cio.load_ranking()
    else:
        raise ValueError("unknown scope {!r}".format(scope))

    # Blank on DEP exactly what ranking.parquet blanks, BEFORE featurising, so a
    # feature cannot read a column that will not exist at scoring time.
    mov = cio.apply_ranking_contract(mov)
    cio.assert_ranking_computable(F.source_columns(featureset),
                                  name="{}/{}".format(scope, featureset))

    X = F.build_features(mov, featureset=featureset)
    X.insert(0, config.ID, mov.loc[X.index, config.ID].to_numpy())
    X.to_parquet(path, index=False)
    _MEM[key] = X
    return X


def _split_ids(split: str, part: str) -> "pd.DataFrame":
    p = config.SPLIT_DIR / split / "{}_ids.parquet".format(part)
    if not p.exists():
        raise FileNotFoundError(
            "{} missing -- run: python3 stage2_split.py --split {}".format(p, split))
    return pd.read_parquet(p)


def _targets() -> "pd.Series":
    """MVT_ID -> taxi-out seconds, for training DEP rows only."""
    if "_y" not in _MEM:
        df = cio.load_training(columns=["MVT_ID_mvt", "PHASE_mvt", config.TARGET])
        dep = df[df.PHASE_mvt.eq("DEP")]
        _MEM["_y"] = pd.Series(dep[config.TARGET].astype("float64").to_numpy(),
                               index=dep[config.ID].to_numpy())
    return _MEM["_y"]


def apply_outlier_policy(y: "pd.Series", policy: str):
    """Return (keep_mask, y_adjusted, info) for a TRAINING target under `policy`.

    Design-gate finding B1. Four shapes, one entry point:
      * impossible values (y <= 0) are dropped under EVERY policy -- corrupt, not tail
      * "drop"      : additionally remove y outside [TRAIN_TARGET_MIN_SEC, bound]
      * "winsorize" : keep every row, clamp the target down to `bound`. Preserves
                      the row (and its features) while bounding its leverage on L2
      * "drop_ids"  : remove specific MVT_IDs (`bound` is a tuple of them) --
                      a named, documented data bug, not a tail-shape bet
      * "keep"      : impossible-value removal only
    """
    if policy not in config.OUTLIER_POLICIES:
        raise ValueError("unknown outlier policy {!r}; known: {}".format(
            policy, sorted(config.OUTLIER_POLICIES)))
    mode, bound = config.OUTLIER_POLICIES[policy]
    y = pd.Series(np.asarray(y, dtype="float64"), index=y.index)

    keep = y > config.TARGET_IMPOSSIBLE_MAX_SEC
    info = {"policy": policy, "mode": mode,
            "bound": bound if mode != "drop_ids" else list(bound),
            "n_in": int(len(y)), "n_impossible_dropped": int((~keep).sum()),
            "n_tail_dropped": 0, "n_winsorized": 0, "n_ids_dropped": 0}

    if mode == "drop":
        tail = keep & ((y < config.TRAIN_TARGET_MIN_SEC) | (y > bound))
        info["n_tail_dropped"] = int(tail.sum())
        keep = keep & ~tail
    elif mode == "winsorize":
        info["n_winsorized"] = int((keep & (y > bound)).sum())
        y = y.clip(upper=bound)
    elif mode == "drop_ids":
        excl = keep & y.index.isin(pd.Index(bound, dtype=y.index.dtype))
        info["n_ids_dropped"] = int(excl.sum())
        keep = keep & ~excl
    info["n_out"] = int(keep.sum())
    return keep, y, info


def pool_fingerprint(ids) -> str:
    """Stable hash of the IDs a model ACTUALLY consumed (design-gate B2).

    ml_common's own docstring warns to pass the consumed IDs, not the ones the
    split handed over -- MISTAKES.md #56 is a pool-equality gate that passed
    while one arm had silently subsampled to 10.6%. Each train.py writes this
    into model.json and compare.py asserts the three agree.
    """
    import hashlib
    h = hashlib.sha256()
    for i in sorted(np.asarray(ids, dtype="float64").tolist()):
        h.update(repr(i).encode())
    return h.hexdigest()[:32]


def load_part(split: str, part: str, featureset: str, *,
              filter_target: bool = False,
              outlier_policy: str = None):
    """Return (X, y, meta) for one split part.

    `filter_target=True` applies the training outlier policy -- use it for
    TRAINING pools only. An evaluation set must be scored exactly as the
    leaderboard would score it, outliers included; filtering it would report a
    number for an easier problem than the one posed.
    """
    ids = _split_ids(split, part)
    Xall = build_scope("train2025", featureset)
    X = Xall.set_index(config.ID).reindex(ids[config.ID].to_numpy())
    if X.isna().all(axis=1).any():
        missing = int(X.isna().all(axis=1).sum())
        raise RuntimeError(
            "{} split ids for {}/{} have no featurised row ({} of {}). The feature "
            "cache is stale -- rerun with force=True.".format(
                missing, split, part, missing, len(ids)))
    y = _targets().reindex(X.index)
    meta = ids.set_index(config.ID).reindex(X.index)

    if filter_target:
        policy = config.DEFAULT_OUTLIER_POLICY if outlier_policy is None else outlier_policy
        keep, y, info = apply_outlier_policy(y, policy)
        k = keep.to_numpy()
        X, y, meta = X[k], y[k], meta[k]
        meta.attrs["outlier_policy"] = info
    return X, y, meta


def load_ranking_features(featureset: str):
    """(X, meta) for the scored DEP rows, in submitting.parquet's own row order."""
    X = build_scope("ranking", featureset)
    rank = cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt",
                                     "MVT_TIME_UTC_mvt"])
    dep = rank[rank.PHASE_mvt.eq("DEP")]
    sub = pd.read_parquet(config.SUBMITTING_PARQUET, columns=[config.ID])
    X = X.set_index(config.ID).reindex(sub[config.ID].to_numpy())
    if X.isna().all(axis=1).any():
        raise RuntimeError("submitting.parquet contains IDs absent from ranking DEP")
    meta = dep.set_index(config.ID).reindex(X.index)
    return X, meta


def align_columns(X_fit: "pd.DataFrame", X_apply: "pd.DataFrame") -> "pd.DataFrame":
    """Reindex X_apply to X_fit's columns and re-apply X_fit's category levels.

    Two frames built by the same code still disagree on category ORDER when they
    contain different level sets; a model trained on one then scored on the other
    silently re-codes every categorical. This makes the codes match or fails.
    """
    out = X_apply.reindex(columns=X_fit.columns)
    for c in X_fit.columns:
        if isinstance(X_fit[c].dtype, pd.CategoricalDtype):
            out[c] = pd.Categorical(out[c], categories=X_fit[c].cat.categories)
    return out
