"""Per-(airport, flight-type-bucket) LightGBM, nested under per-airport routing.

Built to actually test the "airport x timing/flight-type" segmentation idea
that was reasoned against on cardinality grounds earlier this session, rather
than leave it as an untested hypothesis. FLIGHT_TYPE_flt is ~95.3% 'S'
(scheduled) in train, with the remaining ~4.7% split across N/X/G/M/NaN
(charter/other categories, each individually tiny per airport -- e.g. 'M' is
410 rows total across all 10 airports). A full airport x flight_type cross
(10 x 6 = up to 60 cells) would put several cells in the single digits.
Bucketing to S vs Other keeps every one of the 20 (airport, bucket) cells
above ~1,300 rows in train (checked empirically, not assumed), which is the
minimum for this to be a fair test rather than a noise generator.

Each airport still gets its own pooled booster (all flight types) as a
fallback for any (airport, bucket) cell that falls below `min_cell_rows` in a
given fit -- defends against a bucket that is fine in train but thin in a
particular seed's inner split, the same shape of guard as PerAirportLGBM's
unseen-airport fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from pipeline_per_airport.model import AIRPORT_COL

FLIGHT_TYPE_COL = "FLIGHT_TYPE_flt"
DEFAULT_MIN_CELL_ROWS = 1000


def _bucket(X: "pd.DataFrame") -> np.ndarray:
    ft = X[FLIGHT_TYPE_COL].astype("string").fillna("NaN")
    return ft.where(ft == "S", "Other").to_numpy()


class PerAirportFlightTypeLGBM:
    """A booster per (airport, flight-type bucket), pooled-per-airport fallback."""

    def __init__(self, params: dict, num_boost_round: int, patience: int,
                 cats: "list[str]", min_cell_rows: int = DEFAULT_MIN_CELL_ROWS):
        self.params = dict(params)
        self.num_boost_round = int(num_boost_round)
        self.patience = int(patience)
        self.cats = list(cats)
        self.min_cell_rows = int(min_cell_rows)
        self.boosters_: dict = {}          # key "AIRPORT|BUCKET" -> booster
        self.best_iter_: dict = {}
        self.pooled_boosters_: dict = {}   # key "AIRPORT" -> booster (fallback)
        self.pooled_best_iter_: dict = {}
        self.n_train_: dict = {}
        self.convergence_: dict = {}
        self.used_fallback_for_: "list[str]" = []

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

    def fit(self, X, y, meta, Xv=None, yv=None, meta_v=None, *, lgb=None,
            verbose: bool = True):
        ap = meta[AIRPORT_COL].astype("object").to_numpy()
        bk = _bucket(X)
        ap_v = meta_v[AIRPORT_COL].astype("object").to_numpy() if meta_v is not None else None
        bk_v = _bucket(Xv) if Xv is not None else None
        y = np.asarray(y, dtype="float64")

        for a in sorted(pd.unique(ap)):
            m_a = ap == a
            mv_a = (ap_v == a) if ap_v is not None else None

            # pooled-per-airport fallback, always fitted (cheap, and doubles as
            # the airport-level baseline to compare each split against)
            b_pool, best_pool = self._fit_one(
                lgb, X[m_a], y[m_a],
                Xv[mv_a] if mv_a is not None else None,
                np.asarray(yv)[mv_a] if mv_a is not None else None,
                "{}|POOLED".format(a))
            self.pooled_boosters_[str(a)] = b_pool
            self.pooled_best_iter_[str(a)] = best_pool

            for bucket in ("S", "Other"):
                m = m_a & (bk == bucket)
                n = int(m.sum())
                key = "{}|{}".format(a, bucket)
                if n < self.min_cell_rows:
                    self.used_fallback_for_.append(key)
                    if verbose:
                        print("  {:20s} n={:7d} -> below min_cell_rows={}, using "
                              "pooled-per-airport fallback".format(key, n, self.min_cell_rows))
                    continue
                mv = (mv_a & (bk_v == bucket)) if mv_a is not None else None
                b, best = self._fit_one(lgb, X[m], y[m],
                                        Xv[mv] if mv is not None else None,
                                        np.asarray(yv)[mv] if mv is not None else None,
                                        key)
                self.boosters_[key] = b
                self.best_iter_[key] = best
                self.n_train_[key] = n
                if verbose:
                    print("  {:20s} n={:7d} best_iter={:5d} {}".format(
                        key, n, best,
                        "early-stopped" if self.convergence_[key]["early_stop_fired"]
                        else "HIT ROUND CAP"))
        return self

    def predict(self, X, meta) -> np.ndarray:
        ap = meta[AIRPORT_COL].astype("object").to_numpy()
        bk = _bucket(X)
        out = np.full(len(X), np.nan, dtype="float64")
        for a in pd.unique(ap):
            for bucket in ("S", "Other"):
                m = (ap == a) & (bk == bucket)
                if not m.any():
                    continue
                key = "{}|{}".format(a, bucket)
                if key in self.boosters_:
                    out[m] = self.boosters_[key].predict(
                        X[m], num_iteration=self.best_iter_[key])
                elif str(a) in self.pooled_boosters_:
                    out[m] = self.pooled_boosters_[str(a)].predict(
                        X[m], num_iteration=self.pooled_best_iter_[str(a)])
                else:
                    raise RuntimeError(
                        "airport {!r} has neither a specific booster for bucket "
                        "{!r} nor a pooled fallback".format(a, bucket))
        if not np.isfinite(out).all():
            raise RuntimeError("{} non-finite predictions after routing".format(
                int((~np.isfinite(out)).sum())))
        return out
