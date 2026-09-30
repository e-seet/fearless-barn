#!/usr/bin/env python3
"""Full adsb.lol extraction for one day: keep every ADS-B point near the departure airport for each
of our departures, so movement features can be built and rebuilt without re-downloading.

Why a second fetcher: tests_adsb_coverage kept only the first/last ground-sighting time per flight,
which supports one crude feature. That test could not close ADS-B; this one feeds a proper test.

Source: the day's archive listed in adsblol/globe_history_<year>/PREFERRED_RELEASES.txt (the
maintainers' chosen replica, which covers the days whose prod-0 tag 404s). Streamed, never stored.

Per departure (2025: training movements; 2026: ranking movements), keyed on BOTH identifiers
(FLIGHT_mvt and, where present, CALLSIGN_flt): every trace point whose carried-forward callsign
matches, within RADIUS_KM of the airport, in [T - 4 h, T + 10 min], where T = MVT_TIME (takeoff).

Output: points/<day>.parquet with columns
  MVT_ID_mvt, hex, t_rel (s, point time - T), lat, lon, ground (bool), alt_ft, gs_kt, track
and points/<day>.json with counts. Skips a day whose parquet already exists.

usage: python tests_adsb_full/fetch.py 2025-07-13
"""
from __future__ import annotations
import gzip, io, json, math, sys, tarfile, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from common import io as cio

AIRPORTS = {"EGLL": (51.4706, -0.4619), "EHAM": (52.3086, 4.7639), "EDDF": (50.0333, 8.5706),
            "LTFM": (41.2753, 28.7519), "LFPG": (49.0097, 2.5479), "LEMD": (40.4719, -3.5626),
            "LEBL": (41.2971, 2.0785), "LIRF": (41.8003, 12.2389), "EDDM": (48.3538, 11.7861),
            "LSZH": (47.4647, 8.5492)}
RADIUS_KM, WIN_BEFORE, WIN_AFTER = 10.0, 4 * 3600, 600
UA = {"User-Agent": "fearless-barn-prc2026-adsb-extract"}
OUT = HERE / "points"; OUT.mkdir(exist_ok=True)


def preferred_urls(day: str) -> list[str]:
    tag = f"v{day.replace('-', '.')}-planes-readsb-"
    # a year's last day can be published in the next year's repository (2025-12-31 is)
    for year in (day[:4], str(int(day[:4]) + 1)):
        cache = HERE / f"PREFERRED_RELEASES_{year}.txt"
        if not cache.exists():
            u = f"https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt"
            cache.write_bytes(urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read())
        for line in cache.read_text().splitlines():
            urls = [x.strip() for x in line.split(",") if x.strip()]
            if urls and f"/{tag}" in urls[0]:
                return urls
    raise SystemExit(f"{day}: not in PREFERRED_RELEASES of {day[:4]} or the next year")


class HTTPChain(io.RawIOBase):
    """Sequential reader over split archive parts; resumes with Range if the connection drops."""
    def __init__(self, urls):
        self.urls, self.i, self.off, self.resp, self.n = urls, 0, 0, None, 0
        self.total = {}          # part index -> full size from Content-Range
        self.resumes = 0
    def readable(self): return True
    def _open(self):
        req = urllib.request.Request(self.urls[self.i], headers={**UA, "Range": f"bytes={self.off}-"})
        self.resp = urllib.request.urlopen(req, timeout=120)
        cr = self.resp.headers.get("Content-Range")
        if cr and "/" in cr:
            self.total[self.i] = int(cr.rsplit("/", 1)[1])
    def read(self, n=-1):
        n = (1 << 20) if n is None or n < 0 else n
        tries = 0
        while self.i < len(self.urls):
            if self.resp is None:
                try:
                    self._open()
                except Exception:
                    tries += 1
                    if tries > 30: raise
                    time.sleep(min(60, 5 * tries)); continue
            try:
                b = self.resp.read(n)
            except Exception:
                self.resp = None; self.resumes += 1; time.sleep(3); continue
            if b:
                self.off += len(b); self.n += len(b); return b
            # an empty read before the part's full size is a dropped connection, not the end: resume
            full = self.total.get(self.i)
            if full is None or self.off < full:
                self.resp = None; self.resumes += 1; tries += 1
                if tries > 30:
                    raise IOError(f"part {self.i} stuck at {self.off} of {full}")
                time.sleep(min(60, 5 * tries)); continue
            self.i, self.off, self.resp = self.i + 1, 0, None
        return b""
    def complete(self):
        return self.i >= len(self.urls) and len(self.total) == len(self.urls)
    def readinto(self, buf):
        b = self.read(len(buf)); buf[:len(b)] = b; return len(b)


def targets(day: str) -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt"]
    d = cio.load_training(columns=cols) if day.startswith("2025") else cio.load_ranking(columns=cols)
    d = d[d.PHASE_mvt.eq("DEP") & d.ADEP_mvt.isin(AIRPORTS)].copy()
    d["T"] = pd.to_datetime(d.MVT_TIME_UTC_mvt).dt.tz_localize(None).astype("datetime64[ns]").astype("int64") / 1e9
    d0 = pd.Timestamp(day).value / 1e9
    # the archive for `day` holds that UTC day's traces; a takeoff in the first hours of the next day
    # has its pre-takeoff ground points in this file, so those rows are included too (points for one
    # MVT_ID are concatenated across day files downstream)
    d = d[(d["T"] >= d0) & (d["T"] < d0 + 86400 + WIN_BEFORE)].copy()
    return d


def main(day: str) -> int:
    out = OUT / f"{day}.parquet"
    if out.exists():
        print(f"[{day}] exists, skip", flush=True); return 0
    tg = targets(day)
    keys = {}
    for r in tg.itertuples():
        for k in {str(r.FLIGHT_mvt).strip().upper(), str(r.CALLSIGN_flt).strip().upper()}:
            if k and k not in ("NAN", "NONE"):
                keys.setdefault(k, []).append((r.MVT_ID_mvt, r.ADEP_mvt, r.T))
    urls = preferred_urls(day)
    print(f"[{day}] {len(tg)} departures, {len(keys)} keys, parts {[u.rsplit('.', 1)[-1] for u in urls]} ({urls[0].split('/')[-2]})", flush=True)
    t0, rows, nfile, nhit = time.time(), [], 0, 0
    chain = HTTPChain(urls)
    tf = tarfile.open(fileobj=io.BufferedReader(chain, 1 << 20), mode="r|")
    for m in tf:
        if not (m.isfile() and "trace_full" in m.name) or m.size < 200:
            continue
        nfile += 1
        if nfile % 25000 == 0:
            print(f"[{day}] {nfile} files, {nhit} hits, {chain.n/1e9:.2f} GB, {(time.time()-t0)/60:.1f} min", flush=True)
        try:
            data = tf.extractfile(m).read()
            j = json.loads(gzip.decompress(data)) if data[:2] == b"\x1f\x8b" else json.loads(data)
        except Exception:
            continue
        tr, base, hexid = j.get("trace") or [], float(j.get("timestamp", 0)), str(j.get("icao", m.name))
        cur, css = None, []
        for p in tr:
            e = p[8] if len(p) > 8 else None
            if isinstance(e, dict) and e.get("flight"):
                cur = e["flight"].strip().upper()
            css.append(cur)
        hit = {c for c in set(css) if c in keys}
        if not hit:
            continue
        nhit += 1
        for cs in hit:
            for mvt, ap, T in keys[cs]:
                lat0, lon0 = AIRPORTS[ap]
                coslat = math.cos(math.radians(lat0))
                for p, c in zip(tr, css):
                    if c != cs:
                        continue
                    t = base + p[0]
                    if t < T - WIN_BEFORE or t > T + WIN_AFTER:
                        continue
                    if p[1] is None or p[2] is None:
                        continue
                    dx = (p[2] - lon0) * 111.32 * coslat; dy = (p[1] - lat0) * 110.57
                    if dx * dx + dy * dy > RADIUS_KM ** 2:
                        continue
                    alt = p[3]
                    rows.append((mvt, hexid, t - T, p[1], p[2], alt == "ground",
                                 np.nan if alt == "ground" or alt is None else float(alt),
                                 np.nan if len(p) < 5 or p[4] is None else float(p[4]),
                                 np.nan if len(p) < 6 or p[5] is None else float(p[5])))
    # the tar reader stops quietly on a corrupt header; make sure the whole archive was consumed
    while chain.read(1 << 20):
        pass
    expected = sum(chain.total.values())
    if not chain.complete() or chain.n < expected or nfile < 20000:
        print(f"[{day}] INCOMPLETE: read {chain.n/1e9:.2f} of {expected/1e9:.2f} GB, {nfile} trace files; nothing written", flush=True)
        return 2
    df = pd.DataFrame(rows, columns=["MVT_ID_mvt", "hex", "t_rel", "lat", "lon", "ground", "alt_ft", "gs_kt", "track"])
    df.to_parquet(out, index=False)
    stats = {"day": day, "departures": int(len(tg)), "departures_with_points": int(df.MVT_ID_mvt.nunique()),
             "points": int(len(df)), "files": nfile, "hits": nhit, "gb": chain.n / 1e9,
             "minutes": (time.time() - t0) / 60, "release": urls[0].split("/")[-2]}
    (OUT / f"{day}.json").write_text(json.dumps(stats))
    print(f"[{day}] done: {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
