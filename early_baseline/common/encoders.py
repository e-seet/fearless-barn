"""Leak-free encoding for the linear arm.

High-cardinality categoricals (1,899 stands, 6,345 stand-runway pairs) cannot be
one-hot encoded into a ridge design matrix at 1.4M rows. Target encoding is the
standard answer and the standard way to leak: an encoding fitted on the same rows
the model then trains on lets the model read each row's own target back out of
its features.

The recipe here is the leak-free one:
  * TRAIN rows get OUT-OF-FOLD encodings -- each row's category mean is computed
    from the K-1 folds that exclude it.
  * VAL / TEST / RANKING rows get the FULL-TRAIN encoding (they contributed
    nothing to it, so there is nothing to hold out).
  * Unseen levels and small levels fall back to the global mean via smoothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class OutOfFoldTargetEncoder:
    def __init__(self, cols, *, n_splits: int = 5, smoothing: float = 20.0,
                 seed: int = 0):
        self.cols = list(cols)
        self.n_splits = n_splits
        self.smoothing = float(smoothing)
        self.seed = seed
        self.maps_: dict = {}
        self.prior_: float = float("nan")

    def _smoothed(self, s: "pd.Series", y: "pd.Series") -> "pd.Series":
        g = y.groupby(s.astype("object"), observed=True).agg(["sum", "count"])
        return (g["sum"] + self.prior_ * self.smoothing) / (g["count"] + self.smoothing)

    def fit_transform_train(self, X: "pd.DataFrame", y: "pd.Series") -> "pd.DataFrame":
        y = pd.Series(np.asarray(y, dtype="float64"), index=X.index)
        self.prior_ = float(y.mean())
        out = pd.DataFrame(index=X.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        for c in self.cols:
            enc = pd.Series(np.nan, index=X.index, dtype="float64")
            for tr_i, va_i in kf.split(X):
                m = self._smoothed(X[c].iloc[tr_i], y.iloc[tr_i])
                enc.iloc[va_i] = X[c].iloc[va_i].astype("object").map(m).to_numpy()
            out["te_" + c] = enc.fillna(self.prior_)
            # the full-train map is what val/test/ranking will use
            self.maps_[c] = self._smoothed(X[c], y)
        return out

    def transform(self, X: "pd.DataFrame") -> "pd.DataFrame":
        if not self.maps_:
            raise RuntimeError("transform() before fit_transform_train()")
        out = pd.DataFrame(index=X.index)
        for c in self.cols:
            out["te_" + c] = (X[c].astype("object").map(self.maps_[c])
                              .astype("float64").fillna(self.prior_))
        return out
