#!/usr/bin/env python3
"""Addendum 4: the frozen shipped configuration, trained on the 18 old days, scored on fresh days."""
import sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
FRESH = set((HERE / "fresh_days.txt").read_text().split())
src = (HERE / "ship_config.py").read_text(encoding="utf-8")
# same code as ship_config.py, with the fresh days added to the loaded set and the split changed to old -> fresh
src = src.replace('for d in JAN + JUL:', 'for d in JAN + JUL + sorted(FRESH):')
src = src.replace('days = {d for d in JAN + JUL if', 'days = {d for d in JAN + JUL + sorted(FRESH) if')
src = src.replace('tr = df[(df.mon == a) & df.ap.isin(SHIP)]', 'tr = df[(df.mon == a) & df.ap.isin(SHIP) & ~df.day.isin(FRESH)]')
src = src.replace('te = df[df.mon == b].copy()', 'te = df[(df.mon == b) & df.day.isin(FRESH)].copy()')
src = src.replace('"ship_config.json"', '"confirm.json"')
assert src.count("FRESH") >= 4
src += '''
ok = all(out[k]["gain"] < 0 and out[k]["adsb"] < min(out[k]["perm"]) and
         sum(v < 0 for v in out[k]["per_day"].values()) >= 2 / 3 * len(out[k]["per_day"]) for k in out)
print(f"CONFIRMATION (addendum 4): {'PASS' if ok else 'FAIL'}")
'''
exec(compile(src, "confirm", "exec"), {"__file__": str(HERE / "ship_config.py"), "__name__": "__main__", "FRESH": FRESH})
