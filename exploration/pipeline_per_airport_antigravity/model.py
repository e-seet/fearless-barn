"""Per-airport LightGBM model with stand-runway geometry priors and localized fallback.

Trained per departure airport (ADEP) so that each airport's local runway geometry,
stand clusters, taxiway topologies, and traffic queues converge at their own optimal depth and iterations.
"""
from __future__ import annotations

import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "exploration")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

AIRPORT_COL = "ADEP_mvt"


class PerAirportLGBMAntigravity:
    """Specialized per-airport LightGBM models with fallback."""

    def __init__(self, params: dict, num_boost_round: int, patience: int, cats: list[str]):
        self.params = dict(params)
        self.num_boost_round = int(num_boost_round)
        self.patience = int(patience)
        self.cats = list(cats)
        self.boosters_: dict = {}
        self.best_iter_: dict = {}
        self.n_train_: dict = {}
        self.convergence_: dict = {}
        self.fallback_ = None
        self.fallback_best_iter_ = None

    @staticmethod
    def _airports(meta: pd.DataFrame) -> np.ndarray:
        return meta[AIRPORT_COL].astype("object").to_numpy()

    def _fit_one(self, lgb, X, y, Xv, yv, tag: str):
        cats_here = [c for c in self.cats if c in X.columns]
        dtr = lgb.Dataset(X, label=np.asarray(y, dtype="float64"),
                          categorical_feature=cats_here, free_raw_data=False)
        callbacks = [lgb.log_evaluation(0)]
        valid_sets, valid_names = None, None
        if Xv is not None and len(Xv):
            dva = lgb.Dataset(Xv, label=np.asarray(yv, dtype="float64"),
                              categorical_feature=cats_here, reference=dtr,
                              free_raw_data=False)
            valid_sets, valid_names = [dva], ["val"]
            callbacks.append(lgb.early_stopping(self.patience, verbose=False))
        booster = lgb.train(self.params, dtr,
                            num_boost_round=self.num_boost_round,
                            valid_sets=valid_sets, valid_names=valid_names,
                            callbacks=callbacks)
        best = booster.best_iteration or self.num_boost_round
        hit_cap = bool(best >= self.num_boost_round)
        self.convergence_[tag] = {
            "best_iteration": int(best),
            "max_rounds": self.num_boost_round,
            "patience": self.patience,
            "early_stop_fired": bool(valid_sets is not None and not hit_cap),
            "hit_max_rounds": hit_cap,
        }
        return booster, int(best)

    def fit(self, X, y, meta, Xv=None, yv=None, meta_v=None, *, lgb=None,
            frozen: dict = None, verbose: bool = True):
        ap = self._airports(meta)
        ap_v = self._airports(meta_v) if meta_v is not None else None
        for a in sorted(pd.unique(ap)):
            m = ap == a
            mv = (ap_v == a) if ap_v is not None else None
            if frozen is not None:
                self.num_boost_round = int(frozen["best_iteration_per_airport"][str(a)])
                b, best = self._fit_one(lgb, X[m], np.asarray(y)[m], None, None, str(a))
            else:
                b, best = self._fit_one(lgb, X[m], np.asarray(y)[m],
                                        Xv[mv] if mv is not None else None,
                                        np.asarray(yv)[mv] if mv is not None else None,
                                        str(a))
            self.boosters_[str(a)] = b
            self.best_iter_[str(a)] = best
            self.n_train_[str(a)] = int(m.sum())
            if verbose:
                print("  {:6s} n={:7d} best_iter={:5d} {}".format(
                    str(a), int(m.sum()), best,
                    "(frozen)" if frozen is not None else
                    ("early-stopped" if self.convergence_[str(a)]["early_stop_fired"]
                     else "HIT ROUND CAP")))
        return self

    def predict(self, X, meta) -> np.ndarray:
        ap = self._airports(meta)
        out = np.full(len(X), np.nan, dtype="float64")
        for a in pd.unique(ap):
            m = ap == a
            key = str(a)
            if key in self.boosters_:
                out[m] = self.boosters_[key].predict(
                    X[m], num_iteration=self.best_iter_[key])
            elif self.fallback_ is not None:
                out[m] = self.fallback_.predict(X[m])
            else:
                raise RuntimeError(
                    f"airport {key!r} has no per-airport booster and no fallback fitted")
        return np.clip(out, config.PREDICTION_CLIP_MIN_SEC, config.PREDICTION_CLIP_MAX_SEC)
