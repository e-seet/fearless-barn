"""TwoStageDelayModel: hurdle architecture decomposing taxi-out into nominal duration + expected delay.

Architecture:
1. Baseline lookup b_i:
   Unimpeded taxi time computed hierarchically:
   - Median taxi duration per stand_runway (ADEP|STAND|RUNWAY)
   - Fallback to median taxi duration per adep_runway (ADEP|RUNWAY)
   - Fallback to median taxi duration per ADEP
   - Global fallback median

2. Residual delay definition:
   D_i = max(0, y_i - b_i)
   is_delayed_i = (D_i > delay_threshold)

3. Stage 1 (Classifier):
   Predicts P(D_i > delay_threshold | X) using binary cross-entropy (LightGBM).

4. Stage 2 (Conditional Regressor):
   Predicts E[D_i | is_delayed, X] trained on delayed instances using L2 loss (LightGBM).

5. Joint inference:
   y_hat = b_i + P(is_delayed | X) * E[D_i | is_delayed, X]
   clipped to [config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC].
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import config


class TwoStageDelayModel:
    def __init__(self, delay_threshold_sec: float = 180.0):
        self.delay_threshold_sec = float(delay_threshold_sec)
        self.clf: lgb.Booster | None = None
        self.reg: lgb.Booster | None = None
        self.baseline_stand_rwy: dict[str, float] = {}
        self.baseline_adep_rwy: dict[str, float] = {}
        self.baseline_adep: dict[str, float] = {}
        self.baseline_global: float = 900.0

    def fit_baselines(self, X: pd.DataFrame, y: np.ndarray | pd.Series):
        """Fit hierarchical unimpeded baselines from training data."""
        df = pd.DataFrame({
            "y": np.asarray(y, dtype="float64"),
            "adep": X["ADEP_mvt"].astype(str),
            "stand_runway": X["stand_runway"].astype(str) if "stand_runway" in X.columns else X["ADEP_mvt"].astype(str),
            "adep_runway": X["adep_runway"].astype(str) if "adep_runway" in X.columns else X["ADEP_mvt"].astype(str),
        })

        # Calculate robust nominal medians
        self.baseline_global = float(df["y"].median())
        self.baseline_adep = df.groupby("adep")["y"].median().to_dict()
        self.baseline_adep_rwy = df.groupby("adep_runway")["y"].median().to_dict()
        self.baseline_stand_rwy = df.groupby("stand_runway")["y"].median().to_dict()

    def get_baselines(self, X: pd.DataFrame) -> np.ndarray:
        """Lookup hierarchical nominal duration for each row."""
        n = len(X)
        b = np.full(n, self.baseline_global, dtype="float64")

        adep = X["ADEP_mvt"].astype(str).to_numpy()
        for idx, val in enumerate(adep):
            if val in self.baseline_adep:
                b[idx] = self.baseline_adep[val]

        if "adep_runway" in X.columns:
            ar = X["adep_runway"].astype(str).to_numpy()
            for idx, val in enumerate(ar):
                if val in self.baseline_adep_rwy:
                    b[idx] = self.baseline_adep_rwy[val]

        if "stand_runway" in X.columns:
            sr = X["stand_runway"].astype(str).to_numpy()
            for idx, val in enumerate(sr):
                if val in self.baseline_stand_rwy:
                    b[idx] = self.baseline_stand_rwy[val]

        return b

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Joint hurdle prediction."""
        if self.clf is None or self.reg is None:
            raise RuntimeError("Model is not fitted. Train or load boosters first.")

        b = self.get_baselines(X)
        p_delay = np.clip(self.clf.predict(X), 0.0, 1.0)
        d_pred = np.maximum(0.0, self.reg.predict(X))

        pred = b + (p_delay * d_pred)
        return np.clip(pred, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
