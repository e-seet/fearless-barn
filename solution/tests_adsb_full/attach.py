"""Attach the all_nd ADS-B features to a feature frame (PREREGISTERED_tree.md).

Rows are matched on the frame's index (MVT_ID_mvt). Values are NaN outside the 7 corrected airports and
where no strict-matched track exists. Estimates come from every fetched day (points/*.parquet via the
est_cache), so the same function serves 2025 training rows and 2026 ranking rows.
"""
from __future__ import annotations
import sys as _s, pathlib as _p; _r = _p.Path(__file__).resolve().parents[2]; exec(open(_r / "tools" / "repo_layout.py").read()); dual(_r / "solution")
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features

ADSB_AIRPORTS = ["EDDF", "EHAM", "LEBL", "EDDM", "LSZH", "LEMD", "EGLL"]
ADSB_COLS = ["e_first", "e_disp", "e_move", "liftoff_dt", "stop_s", "lineup_s", "med_gs"]
_EST = None


def estimates() -> pd.DataFrame:
    global _EST
    if _EST is None:
        days = sorted(f.stem for f in (HERE / "points").glob("*.parquet"))
        e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt")
        e = e[~e.index.duplicated()]
        e.index = e.index.astype("int64")
        _EST = e[ADSB_COLS]
        print(f"[adsb] estimates from {len(days)} days ({days[0]} .. {days[-1]}), {len(_EST)} matched departures", flush=True)
    return _EST


def attach(X: pd.DataFrame) -> pd.DataFrame:
    e = estimates().reindex(pd.Index(X.index).astype("int64"))
    ap = X["ADEP_mvt"].astype(str).to_numpy()
    out = X.copy()
    off = ~np.isin(ap, ADSB_AIRPORTS)
    for c in ADSB_COLS:
        v = e[c].to_numpy(dtype="float64").copy()
        v[off] = np.nan
        out["adsb_" + c] = v
    return out
