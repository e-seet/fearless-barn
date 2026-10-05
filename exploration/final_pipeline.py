#!/usr/bin/env python3
"""End-to-end recipe of the final submission, for ANY movements file in ranking.parquet's schema.

Final board: v56 = 271.5625. Stages (each reads only the given movements file):

  1. base     v42 ensemble (trees + 18 MLPs + 21 FT-Transformers + TabNet override at LIRF), `predict_new.py`
              on the movements file (or a saved base file via --base)
  2. ADS-B    corrector (tests_adsb_full/model_v57: 1500 trees, 63 leaves; v56 used model_v48), 5 seeds averaged, at EDDF, EHAM, LEBL, EDDM, LSZH, LEMD,
              EGLL, on departures with a strict-matched adsb.lol ground track
  3. weather  v49: +30 s, January, no ADS-B correction, METAR frost (tmpf <= 34 F and tmpf - dwpf <= 3 F)
              v50: +200 s, July, no ADS-B correction, METAR thunderstorm (TS in wxcodes)
              METAR = last report at or before takeoff within 90 min; "no ADS-B correction" = stage-2 output equals
              the stage-1 base (the rule the uploaded files used)
  4. LIRF     v55: LIRF corrector (tests_lirf_adsb/model_v55) on January LIRF departures with an NM record
              (FLIGHT_ID_mvt) and a strict track; corrector input `base` = stage-1 value; correction added
  5. LFPG     v56: LFPG corrector (tests_round9/model_v56) on LFPG departures with a record and a strict track

Data it needs, fetched automatically with --fetch when missing:
  ADS-B    one adsb.lol archive per UTC day of the departures (+ the day before, for takeoffs in the first hours),
           streamed by tests_adsb_full/fetch.py into --points-dir (only matched points are kept)
  METAR    prc-2026-datasets-wx/<station>.csv; reports outside it are fetched from the Iowa Environmental Mesonet
           (the same ASOS source) into wx_cache/

Only stage 1 needs the competition's saved base models; stages 2-5 need the saved correctors listed above.

usage:
  # the competition's ranking set, from the saved base, verified against the upload:
  python final_pipeline.py --base submissions/fearless_barn_v42.parquet --check submissions/fearless_barn_v56.parquet --out final.parquet
  # any new movements file, everything from scratch (downloads ADS-B and METAR as needed):
  python final_pipeline.py --movements new.parquet --points-dir adsb_new --fetch --out pred.parquet
"""
import argparse, filecmp, json, subprocess, sys, tempfile, urllib.request
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "tests_adsb_full"))
import config
import features

ADSB = ["e_first", "e_disp", "e_move", "liftoff_dt", "stop_s", "lineup_s", "med_gs"]
WX = HERE.parent / "prc-2026-datasets-wx"
WX_CACHE = HERE / "wx_cache"
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}"
       + "".join(f"&data={c}" for c in ["tmpf", "dwpf", "relh", "sknt", "gust", "vsby", "skyc1", "skyl1", "wxcodes", "p01i", "mslp"])
       + "&year1={y1}&month1={m1}&day1={d1}&year2={y2}&month2={m2}&day2={d2}&tz=Etc%2FUTC&format=onlycomma"
       + "&latlon=no&elev=no&missing=M&trace=T&direct=no&report_type=3&report_type=4")


def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")


# ---------------------------------------------------------------- METAR
def metar_raw(station, t_min, t_max, fetch):
    """Raw METAR rows (strings) covering [t_min - 2 h, t_max]: the local file, plus IEM for any gap."""
    lo, hi = t_min - pd.Timedelta("2h"), t_max
    parts, have = [], None
    if (WX / f"{station}.csv").exists():
        d = pd.read_csv(WX / f"{station}.csv", dtype=str)
        parts.append(d)
        v = pd.to_datetime(d["valid"], utc=True, errors="coerce")
        have = (v.min(), v.max())
    # a gap only matters if some takeoff has no report within the 90-min join window
    gaps = [(lo, hi)] if have is None else ([(lo, min(hi, have[0]))] if t_min < have[0] else []) +         ([(max(lo, have[1]), hi)] if hi > have[1] + pd.Timedelta("90min") else [])
    for g0, g1 in gaps:
        f = WX_CACHE / f"{station}_{g0:%Y%m%d}_{g1:%Y%m%d}.csv"
        if not f.exists():
            if not fetch:
                raise SystemExit(f"METAR for {station} {g0:%Y-%m-%d}..{g1:%Y-%m-%d} is not available locally; rerun with --fetch")
            e = g1 + pd.Timedelta("1D")
            u = IEM.format(st=station, y1=g0.year, m1=g0.month, d1=g0.day, y2=e.year, m2=e.month, d2=e.day)
            WX_CACHE.mkdir(exist_ok=True)
            f.write_bytes(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "fearless-barn"}), timeout=120).read())
            print(f"[metar] fetched {station} {g0:%Y-%m-%d}..{g1:%Y-%m-%d} from IEM", flush=True)
        parts.append(pd.read_csv(f, dtype=str))
    d = pd.concat(parts, ignore_index=True).drop_duplicates("valid", keep="first")
    d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    return d


def latest_metar(ids, times, aps, raws, cols, dropna):
    """Last METAR at or before each takeoff within 90 min, per station (the exact join of build_v49/v50)."""
    frames = []
    for st in sorted(pd.unique(aps)):
        d = raws[st]
        f = pd.DataFrame({"valid": d["valid"], "station": st, **{k: fn(d) for k, fn in cols.items()}})
        frames.append(f.dropna(subset=["valid"] + dropna).sort_values("valid"))
    wx = pd.concat(frames, ignore_index=True).sort_values("valid")
    left = pd.DataFrame({"id": ids, "valid": times, "station": aps}).sort_values("valid")
    return pd.merge_asof(left, wx, on="valid", by="station", direction="backward",
                         tolerance=pd.Timedelta("90min")).set_index("id").reindex(ids)


def boosters(mdir, prefix):
    mc = json.loads((mdir / "config.json").read_text())
    if "seeds" in mc:
        return mc, [lgb.Booster(model_file=str(mdir / f"{prefix}_seed{s}.txt")) for s in mc["seeds"]]
    return mc, [lgb.Booster(model_file=str(mdir / f"{prefix}.txt"))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movements", type=Path, default=config.RANKING_PARQUET)
    ap.add_argument("--base", type=Path, default=None, help="saved stage-1 predictions; default: run predict_new.py")
    ap.add_argument("--ids", type=Path, default=None, help="template whose IDs/order the output follows")
    ap.add_argument("--points-dir", type=Path, default=None, help="ADS-B day files; default tests_adsb_full/points")
    ap.add_argument("--fetch", action="store_true", help="download missing ADS-B days and METAR reports")
    ap.add_argument("--stage2-model", type=Path, default=HERE / "tests_adsb_full" / "model_v57",
                    help="7-airport ADS-B corrector; model_v57 (final, round 10 K) or model_v48 (as in v56)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, default=None)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())

    # ---- stage 1: base
    if a.base is None:
        a.base = tmp / "base.parquet"
        cmd = [sys.executable, str(HERE / "predict_new.py"), "--movements", str(a.movements), "--out", str(a.base)]
        if a.ids: cmd += ["--ids", str(a.ids)]
        subprocess.run(cmd, check=True)
    base = pd.read_parquet(a.base)
    ids = base[config.ID].to_numpy(); ids64 = base[config.ID].astype("int64").to_numpy()

    # ---- row facts from the movements file only
    mv = pd.read_parquet(a.movements, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"])
    mv = mv[mv.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt")
    mv = mv.set_index(mv.MVT_ID_mvt.astype("int64")).reindex(ids64)
    assert mv.ADEP_mvt.notna().all(), "base IDs missing from the movements file"
    t = pd.to_datetime(mv.MVT_TIME_UTC_mvt, utc=True)
    aps = mv.ADEP_mvt.astype(str).to_numpy(); mon = t.dt.month.to_numpy()
    rec = mv.FLIGHT_ID_mvt.notna().to_numpy()

    # ---- ADS-B day files: a takeoff's ground points are in its own day's archive, or the previous day's when it
    #      takes off in the first 4 h (fetch.targets' window), so both are read
    pdir = a.points_dir or (HERE / "tests_adsb_full" / "points")
    days = sorted(set(t.dt.strftime("%Y-%m-%d").dropna()) | set((t - pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%d").dropna()))
    if a.fetch:
        for d in [d for d in days if not (pdir / f"{d}.parquet").exists()]:
            subprocess.run([sys.executable, str(HERE / "tests_adsb_full" / "fetch.py"), d, "--movements", str(a.movements),
                            "--points-dir", str(pdir)], check=True)
    have = [d for d in days if (pdir / f"{d}.parquet").exists()]
    print(f"[adsb] {len(have)} of {len(days)} day files present in {pdir}", flush=True)
    te = pd.DataFrame({"base": base[config.TARGET].astype(float).to_numpy(), "ap": aps, "hour": t.dt.hour.to_numpy()}, index=ids64)
    if have:
        est = pd.concat([features.load_day(d, None if a.points_dir is None else pdir) for d in have]).sort_values("liftoff_dt")
        est = est[~est.index.duplicated()]; est.index = est.index.astype("int64")
        te = te.join(est[ADSB].astype(float))     # an empty day file yields untyped columns; keep them numeric
    else:
        for c in ADSB:
            te[c] = np.nan

    # ---- stage 2: the 7-airport corrector (v48)
    mc, bs = boosters(a.stage2_model, "adsb_corrector")
    ship = mc["airports"]
    cov = (te.e_first.notna() & te.ap.isin(ship)).to_numpy()
    te["apc"] = pd.Categorical(te.ap.where(te.ap.isin(ship)), categories=ship)
    corr = np.zeros(len(te))
    if cov.any():
        corr[cov] = np.mean([b.predict(te.loc[cov, mc["features"]]) for b in bs], axis=0)
    adsb = np.clip(np.rint(te.base.to_numpy() + corr), 0, 86400).astype("int32")
    no_adsb = base[config.TARGET].to_numpy() == adsb

    # ---- stage 3: weather offsets (v49 frost, v50 thunderstorm)
    raws = {st: metar_raw(st, t.min(), t.max(), a.fetch) for st in sorted(set(aps))}
    fw = latest_metar(ids, t.to_numpy(), aps, raws, {"tmpf": lambda d: num(d["tmpf"]), "dwpf": lambda d: num(d["dwpf"])}, ["tmpf", "dwpf"])
    tw = latest_metar(ids, t.to_numpy(), aps, raws, {"is_ts": lambda d: d["wxcodes"].fillna("").str.upper().str.contains("TS")}, [])
    frost = ((fw.tmpf <= 34.0) & ((fw.tmpf - fw.dwpf) <= 3.0)).to_numpy()
    ts = tw.is_ts.fillna(False).to_numpy(dtype=bool)
    pred = adsb.astype(np.float64)
    m49 = (mon == 1) & no_adsb & frost
    pred[m49] += 30.0
    pred = np.clip(np.rint(pred), 60, 86400)
    m50 = (mon == 7) & no_adsb & ts
    pred[m50] += 200.0
    pred = np.clip(np.rint(pred), 60, 86400)

    # ---- stages 4-5: single-airport correctors, added to the running prediction
    n45 = {}
    for stage, mdir, prefix, mask in (("LIRF", HERE / "tests_lirf_adsb" / "model_v55", "lirf_corrector", (aps == "LIRF") & (mon == 1)),
                                      ("LFPG", HERE / "tests_round9" / "model_v56", "lfpg_corrector", aps == "LFPG")):
        mc, bs = boosters(mdir, prefix)
        c = mask & rec & te.e_first.notna().to_numpy()
        if c.any():
            cc = np.mean([b.predict(te.loc[c, mc["features"]]) for b in bs], axis=0)
            pred[c] = np.clip(np.rint(pred[c] + cc), 60, 86400)
        n45[stage] = int(c.sum())

    out = base[[config.ID]].copy(); out[config.TARGET] = pred.astype("int32")
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} rows | ADS-B corrected {int((~no_adsb).sum())} | frost +30 s {int(m49.sum())} | "
          f"thunderstorm +200 s {int(m50.sum())} | LIRF {n45['LIRF']} | LFPG {n45['LFPG']}")
    if a.check:
        ref = pd.read_parquet(a.check)
        same = int((ref[config.TARGET].to_numpy() == out[config.TARGET].to_numpy()).sum())
        print(f"CHECK vs {a.check.name}: identical rows {same} / {len(ref)}; bytes identical {filecmp.cmp(a.check, a.out, shallow=False)}")


if __name__ == "__main__":
    main()
