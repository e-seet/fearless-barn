#!/usr/bin/env python3
"""Fetch METAR (ASOS) for the 10 DC2026 departure airports from Iowa State mesonet.

Writes one CSV per station to prc-2026-datasets-wx/<ICAO>.csv plus a coverage
summary. Nothing here touches the feature fingerprint path -- this is an offline
value-check input, not a pipeline dependency.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.error

OUT = Path(__file__).resolve().parents[0] / "wx_raw"
OUT.mkdir(exist_ok=True)
FINAL = Path("C:/Users/Admin/Desktop/eddie/fearless-barn/prc-2026-datasets-wx")
FINAL.mkdir(exist_ok=True)

STATIONS = ["LTFM", "LFPG", "EGLL", "EHAM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]

BASE = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
        "?station={st}&data=tmpf&data=dwpf&data=relh&data=sknt&data=gust"
        "&data=vsby&data=skyc1&data=skyl1&data=wxcodes&data=p01i&data=mslp"
        "&year1=2025&month1=1&day1=1&year2=2026&month2=8&day2=1"
        "&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T"
        "&direct=no&report_type=3&report_type=4")


def fetch(st: str) -> pd.DataFrame:
    url = BASE.format(st=st)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                raw = r.read().decode("utf-8", "replace")
            df = pd.read_csv(io.StringIO(raw))
            return df
        except (urllib.error.URLError, TimeoutError) as e:
            print("  {} attempt {} failed: {}".format(st, attempt + 1, e))
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("giving up on " + st)


def main() -> int:
    rows = []
    for st in STATIONS:
        print("fetching", st, "...")
        df = fetch(st)
        df.to_csv(OUT / (st + ".csv"), index=False)
        df.to_csv(FINAL / (st + ".csv"), index=False)
        if "valid" in df.columns and len(df):
            v = pd.to_datetime(df["valid"], errors="coerce", utc=True)
            in2025 = v.dt.year.eq(2025).sum()
            in2026 = ((v >= "2026-01-01") & (v < "2026-08-01")).sum()
            # median gap between obs, minutes
            gap = v.sort_values().diff().dt.total_seconds().median() / 60.0
            rows.append(dict(station=st, n=len(df), n_2025=int(in2025),
                             n_2026_JanJul=int(in2026),
                             first=str(v.min()), last=str(v.max()),
                             median_gap_min=round(gap, 1)))
        else:
            rows.append(dict(station=st, n=len(df), n_2025=0, n_2026_JanJul=0,
                             first="", last="", median_gap_min=None))
        print("  ->", rows[-1])
        time.sleep(3)

    summ = pd.DataFrame(rows)
    summ.to_csv(FINAL / "_coverage.csv", index=False)
    print("\n=== COVERAGE ===")
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
