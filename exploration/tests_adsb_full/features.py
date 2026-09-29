#!/usr/bin/env python3
"""Per-departure ADS-B estimates from points/<day>.parquet, with strict matching.

Matching: among the aircraft (hex) that carried the flight's callsign, keep only one whose track
changes from ground to airborne within +-TAKEOFF_TOL s of the recorded takeoff T; if several, the
one with the transition closest to T. Everything else (wrong aircraft, reused callsigns, inbound
legs) is dropped.

Estimates, in seconds before takeoff (so directly comparable with the taxi-out label):
  e_first  first ground point of any kind
  e_disp   first ground point more than DISP_M from the first ground position (left the stand)
  e_move   first ground point at >= 3 kt (the earlier crude definition)
plus n_ground and the takeoff-transition offset (liftoff_dt).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAKEOFF_TOL, DISP_M = 120.0, 40.0


def estimates(points: pd.DataFrame) -> pd.DataFrame:
    p = points.sort_values(["MVT_ID_mvt", "hex", "t_rel"])
    out = []
    for (mvt, hexid), s in p.groupby(["MVT_ID_mvt", "hex"], sort=False):
        g = s.ground.to_numpy()
        t = s.t_rel.to_numpy()
        # ground -> air transitions
        tr = np.flatnonzero(g[:-1] & ~g[1:])
        if len(tr) == 0:
            continue
        lift = t[tr + 1]
        j = np.argmin(np.abs(lift))
        if abs(lift[j]) > TAKEOFF_TOL:
            continue
        gs = s[s.ground & (s.t_rel < lift[j])]
        if len(gs) == 0:
            continue
        lat0, lon0 = gs.lat.iloc[0], gs.lon.iloc[0]
        dx = (gs.lon - lon0).to_numpy() * 111320 * np.cos(np.radians(lat0))
        dy = (gs.lat - lat0).to_numpy() * 110570
        far = np.flatnonzero(np.hypot(dx, dy) > DISP_M)
        mv = np.flatnonzero(gs.gs_kt.to_numpy() >= 3)
        out.append((mvt, hexid, abs(lift[j]), len(gs), -gs.t_rel.iloc[0],
                    -gs.t_rel.iloc[far[0]] if len(far) else np.nan,
                    -gs.t_rel.iloc[mv[0]] if len(mv) else np.nan))
    e = pd.DataFrame(out, columns=["MVT_ID_mvt", "hex", "liftoff_dt", "n_ground", "e_first", "e_disp", "e_move"])
    return e.sort_values("liftoff_dt").drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")


def load_day(day: str) -> pd.DataFrame:
    return estimates(pd.read_parquet(HERE / "points" / f"{day}.parquet"))
