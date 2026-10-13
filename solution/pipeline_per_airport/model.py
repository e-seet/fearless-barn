"""Per-airport LightGBM: one booster per ADEP, routed at prediction time.

Defined in an ordinary module, never in the script the orchestrator runs by path,
so a pickled instance records `pipeline_per_airport.model.PerAirportLGBM` rather
than `__main__.PerAirportLGBM` (MISTAKES.md #78).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

AIRPORT_COL = "ADEP_mvt"


class PerAirportLGBM:
    """A booster per airport plus a pooled fallback for unseen airports.

    The fallback is not decoration: routing is by a categorical value, and a
    router with no default silently produces NaN for any level absent from
    training. All ten airports happen to appear in every split here, so the
    fallback never fires -- which is exactly why it would go untested if it were
    only added when something broke.
    """

    def __init__(self, params: dict, num_boost_round: int, patience: int,
                 cats: "list[str]"):
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

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _airports(meta: "pd.DataFrame") -> np.ndarray:
        return meta[AIRPORT_COL].astype("object").to_numpy()

    def _fit_one(self, lgb, X, y, Xv, yv, tag: str):
        dtr = lgb.Dataset(X, label=np.asarray(y, dtype="float64"),
                          categorical_feature=self.cats, free_raw_data=False)
        callbacks = [lgb.log_evaluation(0)]
        valid_sets, valid_names = None, None
        if Xv is not None and len(Xv):
            dva = lgb.Dataset(Xv, label=np.asarray(yv, dtype="float64"),
                              categorical_feature=self.cats, reference=dtr,
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

    # -- API -------------------------------------------------------------
    def fit(self, X, y, meta, Xv=None, yv=None, meta_v=None, *, lgb=None,
            frozen: dict = None, verbose: bool = True):
        ap = self._airports(meta)
        ap_v = self._airports(meta_v) if meta_v is not None else None
        for a in sorted(pd.unique(ap)):
            m = ap == a
            mv = (ap_v == a) if ap_v is not None else None
            if frozen is not None:
                # refit protocol: the round count arrives frozen, no val exists
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

    def fit_fallback(self, X, y, *, lgb, rounds: int):
        """Pooled booster used only for an airport absent from training."""
        dtr = lgb.Dataset(X, label=np.asarray(y, dtype="float64"),
                          categorical_feature=self.cats, free_raw_data=False)
        self.fallback_ = lgb.train(self.params, dtr, num_boost_round=rounds,
                                   callbacks=[lgb.log_evaluation(0)])
        self.fallback_best_iter_ = rounds
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
                    "airport {!r} has no per-airport booster and no fallback was "
                    "fitted -- routing would emit NaN, which scores as a failure"
                    .format(key))
        if not np.isfinite(out).all():
            raise RuntimeError("{} non-finite predictions after routing".format(
                int((~np.isfinite(out)).sum())))
        return out
