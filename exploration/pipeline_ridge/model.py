"""The ridge arm's fitted-state object, in a module that is never `__main__`.

Pickle records a class by module path AND verifies identity on save: a class
defined inside an entry script is `__main__.RidgePipeline`, which no other
process can resolve (and which `__module__`-rewriting cannot repair, because the
imported module then holds a *different* class object). Keeping it here -- an
ordinary importable module -- is what actually makes `model.pkl` loadable from
`evaluate.py`, `stage4_controls.py`, or an interactive session. MISTAKES.md #78.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from common import features as F
from common.encoders import OutOfFoldTargetEncoder

HIGH_CARD = ["stand_runway", "adep_stand", "adep_runway", "STAND_mvt",
             "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt", "dest_region"]
LOW_CARD = ["ADEP_mvt", "RUNWAY_mvt", "WK_TBL_CAT_flt", "MARKET_SEGMENT_flt",
            "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt"]


def numeric_cols(X: "pd.DataFrame") -> "list[str]":
    return [c for c in X.columns
            if c not in F.CATEGORICAL and pd.api.types.is_numeric_dtype(X[c])]


class RidgePipeline:
    """All fitted state in one picklable object -- defined at module level so it
    survives pickling from any entry point (MISTAKES.md #64)."""

    def __init__(self, seed: int, alpha: float):
        self.seed = seed
        self.alpha = alpha
        self.te = OutOfFoldTargetEncoder(HIGH_CARD, seed=seed)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=seed)
        self.num_cols_: list = []
        self.medians_: "pd.Series" = None
        self.onehot_cols_: list = []

    def _design(self, X, te_frame):
        num = X[self.num_cols_].astype("float64").fillna(self.medians_)
        oh = pd.get_dummies(X[LOW_CARD], columns=LOW_CARD, dummy_na=True, dtype="float64")
        oh = oh.reindex(columns=self.onehot_cols_, fill_value=0.0)
        return pd.concat([num, te_frame, oh], axis=1)

    def fit(self, X, y):
        self.num_cols_ = numeric_cols(X)
        self.medians_ = X[self.num_cols_].astype("float64").median()
        oh = pd.get_dummies(X[LOW_CARD], columns=LOW_CARD, dummy_na=True, dtype="float64")
        self.onehot_cols_ = list(oh.columns)
        te = self.te.fit_transform_train(X, y)
        Z = self._design(X, te)
        Z = pd.DataFrame(self.scaler.fit_transform(Z), index=Z.index, columns=Z.columns)
        self.model.fit(Z, np.asarray(y, dtype="float64"))
        self.feature_names_ = list(Z.columns)
        return self

    def predict(self, X):
        Z = self._design(X, self.te.transform(X))
        Z = pd.DataFrame(self.scaler.transform(Z), index=Z.index, columns=Z.columns)
        p = self.model.predict(Z)
        return np.clip(p, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)

