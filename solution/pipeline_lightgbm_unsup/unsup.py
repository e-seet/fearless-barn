"""Unsupervised feature learning for the taxi-out problem.

Nothing here touches the target. Every transform is fitted on the TRAINING pool
only and applied unchanged to val / test / ranking, which is the same discipline
the supervised preprocessing follows (`methodology/common/data-leakage.md` ->
fit-after-split). Two transforms, each aimed at a specific gap this build has
already documented:

1. **Stand/runway co-occurrence embeddings (truncated SVD).**
   `PLAN.md` sec.9 lists "airport layout / stand coordinates" as an open lever
   that could not be taken because *no coordinates are shipped* -- only
   categorical stand and runway identifiers. But layout leaves a trace in the
   data anyway: which stands feed which runways is a fact about apron geometry,
   and it is observable as a co-occurrence count without any label. Factorising
   the stand x runway count matrix gives each stand a dense vector whose
   neighbours are stands that route to the same runways, and each runway a vector
   whose neighbours are runways fed by the same aprons. That is a learned proxy
   for the physical layout the dataset withholds.

   It also addresses a concrete weakness: `stand_runway` has 6,345 levels, and a
   rare level carries almost no evidence for a native categorical split. Its SVD
   coordinates are informed by every other movement through the same stand.

2. **Day-regime clustering (k-means on daily traffic profiles).**
   Taxi-out has day-level regimes -- a disrupted day (weather, industrial action,
   runway works) is slow for every flight at that airport, and nothing in the
   per-row feature set says "today is a bad day". A profile built from the
   congestion columns aggregated per (airport, day) is fully unsupervised, and
   clustering it gives a small categorical the model can condition on.

Fitted on train, applied everywhere: an unseen stand gets the zero vector (which
the SVD basis treats as "no information", not as a specific location), and an
unseen day profile is assigned to its nearest existing centroid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Columns aggregated into the per-(airport, day) profile. All are counts or
# means of OTHER movements around this one -- none involves this row's target.
PROFILE_COLS = [
    "dep_prev_15m", "dep_prev_30m", "dep_prev_60m",
    "dep_next_15m", "dep_next_30m", "dep_next_60m",
    "arr_prev_15m", "arr_prev_30m", "arr_prev_60m",
    "same_rwy_prev_15m", "same_rwy_prev_30m",
    "n_taxiing_at_pushback", "dep_surge_ratio_15_60",
    "arr_taxiin_mean_prev_60m",
]


class UnsupervisedFeatures:
    """Fit on train features only; transform any part. No target, ever.

    Defined in an ordinary module rather than in the entry script, so a pickled
    instance records an importable class path (MISTAKES.md #78).
    """

    def __init__(self, n_components: int = 4, n_day_clusters: int = 6,
                 seed: int = 42):
        self.n_components = int(n_components)
        self.n_day_clusters = int(n_day_clusters)
        self.seed = int(seed)
        self.stand_emb_: dict = {}
        self.runway_emb_: dict = {}
        self.kmeans_ = None
        self.scaler_ = None
        self.profile_cols_: list = []
        self.fitted_ = False

    # ------------------------------------------------------------------ SVD
    def _fit_cooccurrence(self, X: "pd.DataFrame"):
        from sklearn.decomposition import TruncatedSVD

        stands = X["adep_stand"].astype("object").fillna("NA")
        runways = X["adep_runway"].astype("object").fillna("NA")
        tab = pd.crosstab(stands, runways)
        k = int(min(self.n_components, min(tab.shape) - 1))
        if k < 1:
            # Degenerate airport-set: fall back to no embedding rather than
            # inventing dimensions. Recorded so the caller can see it happened.
            self.stand_emb_, self.runway_emb_ = {}, {}
            return 0
        # log1p because the counts are heavily skewed (a busy stand appears
        # thousands of times); raw counts would make the leading component a
        # popularity axis rather than a topology one.
        M = np.log1p(tab.to_numpy(dtype="float64"))
        svd = TruncatedSVD(n_components=k, random_state=self.seed)
        U = svd.fit_transform(M)                       # stands x k
        V = svd.components_.T                          # runways x k
        self.stand_emb_ = {str(s): U[i] for i, s in enumerate(tab.index)}
        self.runway_emb_ = {str(r): V[j] for j, r in enumerate(tab.columns)}
        self.svd_explained_ = float(svd.explained_variance_ratio_.sum())
        return k

    # -------------------------------------------------------------- k-means
    @staticmethod
    def _day_key(X: "pd.DataFrame", meta: "pd.DataFrame") -> np.ndarray:
        return (meta["ADEP_mvt"].astype(str).to_numpy() + "|"
                + X["doy"].astype("Int64").astype(str).to_numpy())

    def _profiles(self, X: "pd.DataFrame", meta: "pd.DataFrame"):
        cols = [c for c in PROFILE_COLS if c in X.columns]
        key = self._day_key(X, meta)
        df = X[cols].copy()
        df["_key"] = key
        g = df.groupby("_key", observed=True)[cols].mean()
        return cols, key, g

    def _fit_day_regimes(self, X: "pd.DataFrame", meta: "pd.DataFrame"):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        cols, _key, g = self._profiles(X, meta)
        self.profile_cols_ = cols
        P = g.to_numpy(dtype="float64")
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler_ = StandardScaler().fit(P)
        self.kmeans_ = KMeans(n_clusters=self.n_day_clusters, random_state=self.seed,
                              n_init=10).fit(self.scaler_.transform(P))
        return len(g)

    # --------------------------------------------------------------- public
    def fit(self, X: "pd.DataFrame", meta: "pd.DataFrame"):
        k = self._fit_cooccurrence(X)
        n_days = self._fit_day_regimes(X, meta)
        self.fitted_ = True
        print("  [unsup] stand/runway SVD: {} stands x {} runways -> {} dims "
              "({:.1%} variance); day regimes: {} (airport,day) profiles over {} "
              "columns -> {} clusters".format(
                  len(self.stand_emb_), len(self.runway_emb_), k,
                  getattr(self, "svd_explained_", float("nan")), n_days,
                  len(self.profile_cols_), self.n_day_clusters))
        return self

    def transform(self, X: "pd.DataFrame", meta: "pd.DataFrame") -> "pd.DataFrame":
        if not self.fitted_:
            raise RuntimeError("transform() before fit()")
        out = X.copy()
        k = self.n_components if self.stand_emb_ else 0
        if k:
            zero = np.zeros(len(next(iter(self.stand_emb_.values()))))
            S = np.vstack([self.stand_emb_.get(str(v), zero)
                           for v in X["adep_stand"].astype("object").fillna("NA")])
            zero_r = np.zeros(len(next(iter(self.runway_emb_.values()))))
            R = np.vstack([self.runway_emb_.get(str(v), zero_r)
                           for v in X["adep_runway"].astype("object").fillna("NA")])
            for i in range(S.shape[1]):
                out["stand_svd_{}".format(i)] = S[:, i]
            for i in range(R.shape[1]):
                out["runway_svd_{}".format(i)] = R[:, i]

        cols, key, g = self._profiles(X, meta)
        # Use the columns the fit saw, in that order; a part missing one of them
        # would otherwise silently shift every k-means dimension.
        P = g.reindex(columns=self.profile_cols_).to_numpy(dtype="float64")
        P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
        Z = self.scaler_.transform(P)
        lab = pd.Series(self.kmeans_.predict(Z), index=g.index)
        dist = pd.Series(self.kmeans_.transform(Z).min(axis=1), index=g.index)
        out["day_regime"] = pd.Categorical(
            pd.Series(key, index=X.index).map(lab).astype("float64").astype("Int64")
            .astype(str), categories=[str(i) for i in range(self.n_day_clusters)])
        out["day_regime_dist"] = pd.Series(key, index=X.index).map(dist).astype("float64")
        return out

    @property
    def added_columns(self) -> "list[str]":
        cols = []
        if self.stand_emb_:
            k = len(next(iter(self.stand_emb_.values())))
            cols += ["stand_svd_{}".format(i) for i in range(k)]
            k = len(next(iter(self.runway_emb_.values())))
            cols += ["runway_svd_{}".format(i) for i in range(k)]
        return cols + ["day_regime", "day_regime_dist"]
