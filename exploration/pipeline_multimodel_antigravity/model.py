"""MultiModelStackAntigravity: Hybrid stacking model combining per-airport specialist trees with tabular MLP."""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

AIRPORT_COL = "ADEP_mvt"


class MultiModelStackAntigravity:
    def __init__(self, per_airport_model, mlp_model, weights_per_airport: dict[str, float],
                 default_weight: float = 0.5):
        self.per_airport_model = per_airport_model
        self.mlp_model = mlp_model
        self.weights_per_airport = dict(weights_per_airport)
        self.default_weight = float(default_weight)

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame, p_mlp: np.ndarray = None) -> np.ndarray:
        p_airport = self.per_airport_model.predict(X, meta)
        if p_mlp is None:
            raise ValueError("p_mlp predictions must be provided for MultiModelStackAntigravity")
        
        ap = meta[AIRPORT_COL].astype("object").to_numpy()
        p_blend = np.zeros(len(X), dtype="float64")
        
        for a in pd.unique(ap):
            m = (ap == a)
            w = self.weights_per_airport.get(str(a), self.default_weight)
            # w is weight on per_airport model, (1-w) is weight on mlp model
            p_blend[m] = w * p_airport[m] + (1.0 - w) * p_mlp[m]

        return np.clip(p_blend, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
