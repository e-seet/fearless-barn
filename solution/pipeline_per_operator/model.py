"""Per-operator LightGBM: one booster per aircraft operator, routed at prediction time.

Defined in an ordinary module, never in the script the orchestrator runs by path,
so a pickled instance records `pipeline_per_operator.model.PerOperatorLGBM` rather
than `__main__.PerOperatorLGBM` (same reasoning as per_airport, MISTAKES.md #78).

Where this comes from
----------------------
Operator was never tried as a segmentation axis in this build -- only airport
was (per_airport and its variants). A quick isolated check (not this arm --
a throwaway script over the same cached features) found per-operator alone
scores ~277.8 val RMSE, matching per_airport's ~278.2, and that the two are
complementary enough that a plain blend of their predictions beats either
alone. This arm formalises the per-operator half of that finding as a real,
tracked pipeline instead of leaving it in scratch code.

Bucketing, not one booster per raw operator
--------------------------------------------
`AIRCRAFT_OPERATOR_flt` is a high-cardinality, ANONYMISED (hashed) categorical
with a long tail -- unlike ADEP_mvt's ten airports, most operators fly too few
flights to support their own booster. The vocabulary is therefore the top
`top_n` operators BY TRAINING VOLUME (default 20, matching the exploratory
check), with everything else -- rare operators and missing values alike --
routed to one shared "Other" booster. The vocabulary is learned at fit time
only, never from validation or test, so an operator that is rare in train but
common in val/test is still correctly bucketed to "Other" rather than leaking
a val/test-informed vocabulary choice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

OPERATOR_COL = "AIRCRAFT_OPERATOR_flt"
DEFAULT_TOP_N = 20
OTHER_BUCKET = "Other"


class PerOperatorLGBM:
    """A booster per (bucketed) operator plus a pooled fallback for a bucket
    absent from training -- see PerAirportLGBM's docstring for why the
    fallback is not decoration even when it is not expected to fire.
    """

    def __init__(self, params: dict, num_boost_round: int, patience: int,
                 cats: "list[str]", top_n: int = DEFAULT_TOP_N):
        self.params = dict(params)
        self.num_boost_round = int(num_boost_round)
        self.patience = int(patience)
        self.cats = list(cats)
        self.top_n = int(top_n)
        self.vocab_: "list[str]" = None
        self.boosters_: dict = {}
        self.best_iter_: dict = {}
        self.n_train_: dict = {}
        self.convergence_: dict = {}
        self.fallback_ = None
        self.fallback_best_iter_ = None

    # -- helpers ---------------------------------------------------------
    def _bucket(self, meta: "pd.DataFrame") -> np.ndarray:
        raw = meta[OPERATOR_COL].astype(str)  # NaN -> the literal string "nan"
        if self.vocab_ is None:
            raise RuntimeError("fit() must learn the vocabulary before bucketing")
        return raw.where(raw.isin(self.vocab_), OTHER_BUCKET).to_numpy()

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
            frozen: dict = None, vocab: "list[str]" = None, verbose: bool = True):
        if frozen is not None:
            self.vocab_ = list(frozen["vocab"])
        elif vocab is not None:
            self.vocab_ = list(vocab)
        else:
            raw = meta[OPERATOR_COL].astype(str)
            self.vocab_ = raw.value_counts().head(self.top_n).index.tolist()

        op = self._bucket(meta)
        op_v = self._bucket(meta_v) if meta_v is not None else None
        for g in sorted(pd.unique(op)):
            m = op == g
            mv = (op_v == g) if op_v is not None else None
            if frozen is not None:
                self.num_boost_round = int(frozen["best_iteration_per_operator"][g])
                b, best = self._fit_one(lgb, X[m], np.asarray(y)[m], None, None, g)
            else:
                b, best = self._fit_one(lgb, X[m], np.asarray(y)[m],
                                        Xv[mv] if mv is not None else None,
                                        np.asarray(yv)[mv] if mv is not None else None,
                                        g)
            self.boosters_[g] = b
            self.best_iter_[g] = best
            self.n_train_[g] = int(m.sum())
            if verbose:
                print("  {:24s} n={:7d} best_iter={:5d} {}".format(
                    g, int(m.sum()), best,
                    "(frozen)" if frozen is not None else
                    ("early-stopped" if self.convergence_[g]["early_stop_fired"]
                     else "HIT ROUND CAP")))
        return self

    def fit_fallback(self, X, y, *, lgb, rounds: int):
        """Pooled booster used only for a bucket absent from training."""
        dtr = lgb.Dataset(X, label=np.asarray(y, dtype="float64"),
                          categorical_feature=self.cats, free_raw_data=False)
        self.fallback_ = lgb.train(self.params, dtr, num_boost_round=rounds,
                                   callbacks=[lgb.log_evaluation(0)])
        self.fallback_best_iter_ = rounds
        return self

    def predict(self, X, meta) -> np.ndarray:
        op = self._bucket(meta)
        out = np.full(len(X), np.nan, dtype="float64")
        for g in pd.unique(op):
            m = op == g
            if g in self.boosters_:
                out[m] = self.boosters_[g].predict(
                    X[m], num_iteration=self.best_iter_[g])
            elif self.fallback_ is not None:
                out[m] = self.fallback_.predict(X[m])
            else:
                raise RuntimeError(
                    "operator bucket {!r} has no booster and no fallback was "
                    "fitted -- routing would emit NaN, which scores as a failure"
                    .format(g))
        if not np.isfinite(out).all():
            raise RuntimeError("{} non-finite predictions after routing".format(
                int((~np.isfinite(out)).sum())))
        return out
