#!/usr/bin/env python3
"""ADS-B (adsb.lol) coverage check for off-block recovery -- a data pull and a count, no model.

Organizer (relayed 2026-09-21): open trajectory data from which off-block times can be extracted
is allowed; "practically speaking it won't be possible". Opus's discriminating check: for our
AOBT-unmatched rows, how many have an adsb.lol first observation near the gate, BY AIRPORT? If
LIRF is ~0 the lever is dead on coverage alone.

Two things are measured, because coverage is necessary but not sufficient:
  1. COVERAGE  : share of our DEP rows (per airport, matched vs unmatched) whose callsign has >= 1
                 'ground' point inside the airport radius in [takeoff - 4h, takeoff + 5 min].
  2. RECOVERY  : among covered rows, does (first ground observation) reproduce the label's
                 off-block time?  |g0 - BLOCK_TIME| <= 180 s   (BLOCK_TIME is known in training),
                 and for huge-delay rows (y > 3600 s) is the ADS-B taxi time (takeoff - g0)
                 within 30% of y?  This is the decisive test for LIRF's unmatched tail: TAXI_OUT
                 sec.2q / tests_stand_turnover found those labels track MVT-SCHED (a recording
                 artefact), so a physical ADS-B off-block time may DISAGREE with the label even
                 where coverage exists -- in which case it cannot help however good coverage is.

Source per day: prod-0 (~3.3-3.6 GB, all aircraft; staging-0 is a same-size sibling and is not used).
mlatonly-0 (small, multilateration) is tried but was NOT published for the 2025 days checked, so it is
skipped. Streamed over HTTP, never stored (resumes on drop); ~3.5 MB/s => ~17 min per day.

PRE-REGISTERED DECISION RULE (before any download):
  For LIRF unmatched rows (n ~ 72 over the 3 chosen days, ~34 with y > 3600):
   BUILD-WORTHY only if  coverage >= 30%  AND  recovery |g0-BLOCK|<=180s on >= 50% of covered rows
   AND (for covered y>3600 rows) ADS-B taxi within 30% of y on >= 50% of them.
  Otherwise the lever is closed for the population that matters. EDDM is reported as the
  poster's best-case benchmark (~40% claimed).

usage:  python run_adsb_coverage.py fetch 2025-07-13        # one day (run several in parallel)
        python run_adsb_coverage.py analyze                 # after all days are fetched
"""
from __future__ import annotations
import gzip
import io
import json
import math
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT))
import config
from common import io as cio

AIRPORTS = {"EGLL": (51.4706, -0.4619), "EHAM": (52.3086, 4.7639), "EDDF": (50.0333, 8.5706),
            "LTFM": (41.2753, 28.7519), "LFPG": (49.0097, 2.5479), "LEMD": (40.4719, -3.5626),
            "LEBL": (41.2971, 2.0785), "LIRF": (41.8003, 12.2389), "EDDM": (48.3538, 11.7861),
            "LSZH": (47.4647, 8.5492)}
RADIUS_KM = 8.0
WIN_BEFORE, WIN_AFTER = 4 * 3600, 300
UA = {"User-Agent": "claude-code-adsb-coverage-check"}


def hav_km(lat, lon, lat0, lon0):
    p = math.pi / 180
    a = math.sin((lat - lat0) * p / 2) ** 2 + math.cos(lat0 * p) * math.cos(lat * p) * math.sin((lon - lon0) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(a))


def release_assets(day: str, kind: str):
    year = day[:4]
    tag = f"v{day.replace('-', '.')}-planes-readsb-{kind}"
    u = f"https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
    try:
        rel = json.load(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []                      # variant not published for this date (mlatonly-0 only exists for recent days)
        raise
    return sorted(rel["assets"], key=lambda a: a["name"])


class HTTPChain(io.RawIOBase):
    """Sequential reader over split release assets; resumes with Range if the connection drops."""

    def __init__(self, assets):
        self.assets = [(a["browser_download_url"], a["size"]) for a in assets]
        self.i, self.off, self.resp = 0, 0, None

    def readable(self):
        return True

    def _open(self):
        url, size = self.assets[self.i]
        req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={self.off}-"})
        self.resp = urllib.request.urlopen(req, timeout=120)

    def read(self, n=-1):
        if n is None or n < 0:
            n = 1 << 20
        while self.i < len(self.assets):
            if self.resp is None:
                self._open()
            try:
                b = self.resp.read(n)
            except Exception:
                self.resp = None
                time.sleep(3)
                continue
            if b:
                self.off += len(b)
                return b
            self.i, self.off, self.resp = self.i + 1, 0, None
        return b""

    def readinto(self, buf):
        b = self.read(len(buf))
        buf[:len(b)] = b
        return len(b)


def target_rows(day: str) -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
            "TAXITIME_SEC_mvt", "AOBT_3_flt"]
    d = cio.load_training(columns=cols)
    d = d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.isin(AIRPORTS)].copy()
    ep = lambda s: pd.to_datetime(s).dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy() / 1e9
    d["T"] = ep(d.MVT_TIME_UTC_mvt)
    d["B"] = ep(d.BLOCK_TIME_UTC_mvt)
    d0 = pd.Timestamp(day).value / 1e9
    d = d[(d["T"] >= d0 + 2 * 3600) & (d["T"] < d0 + 86400)].copy()     # earlier ground data stays in this day's file
    d["cs"] = d.FLIGHT_mvt.astype(str).str.strip().str.upper()
    d["y"] = d.TAXITIME_SEC_mvt.astype(float)
    d["matched"] = d.AOBT_3_flt.notna()
    return d[["MVT_ID_mvt", "cs", "ADEP_mvt", "T", "B", "y", "matched"]].reset_index(drop=True)


def fetch(day: str):
    tg = target_rows(day)
    by_cs = {}
    for r in tg.itertuples():
        by_cs.setdefault(r.cs, []).append(r)
    print(f"[{day}] target DEP rows {len(tg)} (unmatched {int((~tg.matched).sum())}), distinct callsigns {len(by_cs)}", flush=True)
    out = []
    t0 = time.time()
    for kind in ("mlatonly-0", "prod-0"):
        assets = release_assets(day, kind)
        if not assets:
            print(f"[{day}] {kind}: not published for this date, skipped", flush=True)
            continue
        print(f"[{day}] streaming {kind}: {[a['name'][-3:] + ':' + str(round(a['size'] / 1e6)) + 'MB' for a in assets]}", flush=True)
        tf = tarfile.open(fileobj=HTTPChain(assets), mode="r|")
        nfile = nhit = 0
        for m in tf:
            if not (m.isfile() and "trace_full" in m.name) or m.size < 200:
                continue
            nfile += 1
            if nfile % 20000 == 0:
                print(f"[{day}] {kind}: {nfile} aircraft files scanned, {nhit} target-callsign hits, {(time.time() - t0) / 60:.1f} min", flush=True)
            raw = tf.extractfile(m).read()
            try:
                j = json.loads(gzip.decompress(raw))
            except Exception:
                try:
                    j = json.loads(raw)
                except Exception:
                    continue
            tr, base = j.get("trace") or [], float(j.get("timestamp", 0))
            # current callsign carried forward through the trace
            cur, css = None, []
            for p in tr:
                e = p[8] if len(p) > 8 else None
                if isinstance(e, dict) and e.get("flight"):
                    cur = e["flight"].strip().upper()
                css.append(cur)
            hit = set(c for c in set(css) if c in by_cs)
            if not hit:
                continue
            nhit += 1
            for cs in hit:
                for r in by_cs[cs]:
                    lat0, lon0 = AIRPORTS[r.ADEP_mvt]
                    g = []
                    for p, c in zip(tr, css):
                        if c != cs:
                            continue
                        t = base + p[0]
                        if not (r.T - WIN_BEFORE <= t <= r.T + WIN_AFTER):
                            continue
                        if hav_km(p[1], p[2], lat0, lon0) > RADIUS_KM:
                            continue
                        if p[3] == "ground":
                            g.append(t)
                    out.append(dict(MVT_ID_mvt=r.MVT_ID_mvt, src=kind, n_ground=len(g), g0=min(g) if g else np.nan,
                                    g_last=max(g) if g else np.nan))
        print(f"[{day}] {kind} done: {nfile} aircraft files, {nhit} with a target callsign, {(time.time() - t0) / 60:.1f} min", flush=True)
    res = pd.DataFrame(out)
    if len(res):
        best = res.sort_values("n_ground", ascending=False).drop_duplicates("MVT_ID_mvt")
        tg = tg.merge(best, on="MVT_ID_mvt", how="left")
    else:
        tg["n_ground"], tg["g0"], tg["src"] = 0, np.nan, None
    tg["day"] = day
    tg.to_csv(HERE / f"coverage_{day}.csv", index=False)
    print(f"[{day}] wrote coverage_{day}.csv ({len(tg)} rows)", flush=True)


def analyze():
    d = pd.concat([pd.read_csv(f) for f in sorted(HERE.glob("coverage_*.csv"))], ignore_index=True)
    d["n_ground"] = d.n_ground.fillna(0)
    d["callsign_seen"] = d.src.notna()
    d["covered"] = d.n_ground >= 1
    d["taxi_adsb"] = d["T"] - d.g0
    d["rec_block"] = (d.g0 - d.B).abs() <= 180
    d["big"] = d.y > 3600
    d["rec_big"] = d.big & ((d.taxi_adsb - d.y).abs() <= 0.3 * d.y)
    print(f"days: {sorted(d.day.unique())}  rows {len(d)}\n")
    print("COVERAGE = callsign has a 'ground' point within 8 km of the airport in [T-4h, T+5m]; RECOVERY = |first ground obs - true BLOCK| <= 180 s\n")
    print(f"{'airport':7s} {'group':10s} {'n':>6s} {'callsign seen':>14s} {'covered':>9s} {'recovered':>10s} {'(of covered)':>13s}")
    for ap in ["LIRF"] + [a for a in AIRPORTS if a != "LIRF"]:
        for name, msk in (("unmatched", ~d.matched), ("matched", d.matched)):
            s = d[(d.ADEP_mvt == ap) & msk]
            if not len(s):
                continue
            cov = s.covered
            print(f"{ap:7s} {name:10s} {len(s):6d} {s.callsign_seen.mean():13.0%} {cov.mean():9.0%} {s.rec_block[cov].sum():10d} {(s.rec_block[cov].mean() if cov.any() else float('nan')):13.0%}")
    s = d[(d.ADEP_mvt == "LIRF") & ~d.matched]
    b = s[s.big]
    print(f"\nLIRF unmatched: n={len(s)}, y>3600: n={len(b)}")
    print(f"  coverage {s.covered.mean():.0%} (all) | {b.covered.mean() if len(b) else float('nan'):.0%} (y>3600)")
    cb = b[b.covered]
    print(f"  covered y>3600 rows: {len(cb)}; ADS-B taxi within 30% of y: {int(cb.rec_big.sum())} ({(cb.rec_big.mean() if len(cb) else float('nan')):.0%})")
    if len(cb):
        print("  covered y>3600 examples (y vs ADS-B-derived taxi, s):", [(int(r.y), int(r.taxi_adsb)) for r in cb.head(8).itertuples()])
    cov_ok = s.covered.mean() >= 0.30
    rec_ok = (s.rec_block[s.covered].mean() >= 0.50) if s.covered.any() else False
    big_ok = (cb.rec_big.mean() >= 0.50) if len(cb) else False
    print(f"\nPRE-REGISTERED RULE: coverage>=30% {cov_ok} | recovery>=50% {rec_ok} | big-row agreement>=50% {big_ok}")
    print("VERDICT:", "BUILD-WORTHY" if (cov_ok and rec_ok and big_ok) else "CLOSED for LIRF-unmatched (rule not met)")


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2])
    else:
        analyze()
