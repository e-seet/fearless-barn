#!/usr/bin/env python3
"""End-to-end recipe of the final submission (v56 = 271.5625; stages 1-3 alone give v50 = 271.8748):

  1. base     v42 ensemble (trees + 18 MLPs + 21 FT-Transformers + TabNet override at LIRF)
              -> `predict_new.py` from raw movements, or the saved v42 file (default)
  2. ADS-B    corrector v48: LightGBM on adsb.lol ground-track features, 5 seeds averaged, at EDDF, EHAM, LEBL,
              EDDM, LSZH, LEMD, EGLL, only for flights with a strict-matched track
              -> tests_adsb_full/predict_adsb.py --model-dir tests_adsb_full/model_v48
              (needs tests_adsb_full/points/<day>.parquet for the days predicted: tests_adsb_full/fetch.py <day>)
  3. weather  v49: +30 s, January, no ADS-B correction, METAR frost (tmpf <= 34 F and tmpf - dwpf <= 3 F)
              v50: +200 s, July, no ADS-B correction, METAR thunderstorm (TS in wxcodes)
              METAR = last report at or before takeoff within 90 min (prc-2026-datasets-wx). "No ADS-B correction"
              means the stage-2 output equals the stage-1 base, the rule the uploaded files used.

  4. LIRF     v55: LIRF ADS-B corrector (tests_lirf_adsb/model_v55, 5 seeds) on January LIRF departures with an NM
              record (FLIGHT_ID_mvt) and a strict track; corrector input `base` = stage-1 value; correction added
  5. LFPG     v56: LFPG ADS-B corrector (tests_round9/model_v56, 5 seeds) on LFPG departures with a record and a
              strict track; same convention

The weather offsets were hand-chosen on Jan + Jul 2025 (tests_weather_metar/); an out-of-sample check
(tests_weather_validated/RESULT.md) supports them mildly (-0.07 s pooled; Jan -0.07, Jul -0.31).

usage:
  python final_pipeline.py --out final.parquet --check submissions/fearless_barn_v56.parquet
  python final_pipeline.py --from-raw <movements.parquet> --out final.parquet     # stage 1 from raw data

LIMITATION: stages 2 and 3 look rows up in ranking.parquet (airport, takeoff time via cio.load_ranking /
D.load_ranking_features), and the METAR files end 2026-07-31. --from-raw therefore only works end to end for the
competition's ranking set; a different movements file needs those lookups to read the given file instead.
"""
import argparse, filecmp, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import config
from common import dataset as D

WX = HERE.parent / "prc-2026-datasets-wx"


def num(s):
    return pd.to_numeric(s.replace({"M": np.nan, "T": "0.0001"}), errors="coerce")


def latest_metar(ids, times, aps, cols, dropna):
    """Last METAR at or before each takeoff within 90 min, per station (the exact join of build_v49/v50)."""
    frames = []
    for st in sorted(pd.unique(aps)):
        d = pd.read_csv(WX / f"{st}.csv", dtype=str)
        d["valid"] = pd.to_datetime(d["valid"], utc=True, errors="coerce")
        f = pd.DataFrame({"valid": d["valid"], "station": st, **{k: fn(d) for k, fn in cols.items()}})
        frames.append(f.dropna(subset=["valid"] + dropna).sort_values("valid"))
    wx = pd.concat(frames, ignore_index=True).sort_values("valid")
    left = pd.DataFrame({"id": ids, "valid": times, "station": aps}).sort_values("valid")
    return pd.merge_asof(left, wx, on="valid", by="station", direction="backward",
                         tolerance=pd.Timedelta("90min")).set_index("id").reindex(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", default=None, help="movements parquet; stage 1 runs predict_new.py on it")
    ap.add_argument("--base", default=str(HERE / "submissions" / "fearless_barn_v42.parquet"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp())
    base_path = Path(a.base)
    if a.from_raw:
        base_path = tmp / "base.parquet"
        subprocess.run([sys.executable, str(HERE / "predict_new.py"), "--movements", a.from_raw, "--out", str(base_path)], check=True)
    adsb_path = tmp / "adsb.parquet"
    subprocess.run([sys.executable, str(HERE / "tests_adsb_full" / "predict_adsb.py"), "--model-dir",
                    str(HERE / "tests_adsb_full" / "model_v48"), "--base", str(base_path), "--out", str(adsb_path)], check=True)
    base, adsb = pd.read_parquet(base_path), pd.read_parquet(adsb_path)
    assert base[config.ID].equals(adsb[config.ID])
    no_adsb = base[config.TARGET].to_numpy() == adsb[config.TARGET].to_numpy()

    _, meta = D.load_ranking_features("with_aobt")
    ids = adsb[config.ID].to_numpy()
    meta = meta.reindex(ids)
    t = pd.to_datetime(meta["MVT_TIME_UTC_mvt"], utc=True)
    aps = meta["ADEP_mvt"].astype(str).to_numpy(); mon = t.dt.month.to_numpy()
    frost_wx = latest_metar(ids, t.to_numpy(), aps, {"tmpf": lambda d: num(d["tmpf"]), "dwpf": lambda d: num(d["dwpf"])}, ["tmpf", "dwpf"])
    ts_wx = latest_metar(ids, t.to_numpy(), aps, {"is_ts": lambda d: d["wxcodes"].fillna("").str.upper().str.contains("TS")}, [])
    frost = ((frost_wx.tmpf <= 34.0) & ((frost_wx.tmpf - frost_wx.dwpf) <= 3.0)).to_numpy()
    ts = ts_wx.is_ts.fillna(False).to_numpy(dtype=bool)

    pred = adsb[config.TARGET].to_numpy().astype(np.float64)
    m49 = (mon == 1) & no_adsb & frost
    pred[m49] += 30.0
    pred = np.clip(np.rint(pred), 60, 86400)          # v49 rounding/clipping, then v50 on top
    m50 = (mon == 7) & no_adsb & ts
    pred[m50] += 200.0
    pred = np.clip(np.rint(pred), 60, 86400)                              # = v50

    # stages 4-5: single-airport ADS-B correctors (v55 LIRF in January, v56 LFPG), added to the running prediction
    import lightgbm as lgb
    from common import io as cio
    sys.path.insert(0, str(HERE / "tests_adsb_full"))
    import features
    rk = cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_ID_mvt"])
    rk = rk[rk.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt"); rk = rk.set_index(rk.MVT_ID_mvt.astype("int64"))
    ids64 = adsb[config.ID].astype("int64").to_numpy()
    te = pd.DataFrame({"base": base[config.TARGET].astype(float).to_numpy(), "hour": t.dt.hour.to_numpy(),
                       "rec": rk.FLIGHT_ID_mvt.reindex(ids64).notna().to_numpy()}, index=ids64)
    days = sorted(f.stem for f in (HERE / "tests_adsb_full" / "points").glob("2026-0[17]-*.parquet"))
    e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]
    e.index = e.index.astype("int64")
    te = te.join(e[["e_first", "e_disp", "e_move", "liftoff_dt", "stop_s", "lineup_s", "med_gs"]])
    jan_e = None
    for stage, mdir, prefix, mask in (
            ("LIRF", HERE / "tests_lirf_adsb" / "model_v55", "lirf_corrector", (aps == "LIRF") & (mon == 1)),
            ("LFPG", HERE / "tests_round9" / "model_v56", "lfpg_corrector", aps == "LFPG")):
        mc = json.loads((mdir / "config.json").read_text())
        src = te
        if stage == "LIRF":   # v55 was built from the January 2026 day files only
            if jan_e is None:
                jd = sorted(f.stem for f in (HERE / "tests_adsb_full" / "points").glob("2026-01-*.parquet"))
                jan_e = pd.concat([features.load_day(d) for d in jd]).sort_values("liftoff_dt"); jan_e = jan_e[~jan_e.index.duplicated()]
                jan_e.index = jan_e.index.astype("int64")
            src = te.drop(columns=list(jan_e.columns.intersection(te.columns))).join(jan_e[["e_first", "e_disp", "e_move", "liftoff_dt", "stop_s", "lineup_s", "med_gs"]])
        c = mask & src.rec.to_numpy() & src.e_first.notna().to_numpy()
        bs = [lgb.Booster(model_file=str(mdir / f"{prefix}_seed{sd}.txt")) for sd in mc["seeds"]]
        corr = np.mean([b.predict(src.loc[c, mc["features"]]) for b in bs], axis=0)
        pred[c] = np.clip(np.rint(pred[c] + corr), 60, 86400)
        print(f"stage {stage}: corrected {int(c.sum())}")
    out = adsb.copy(); out[config.TARGET] = pred.astype("int32")
    out.to_parquet(a.out, index=False)
    print(f"wrote {a.out}: ADS-B corrected {int((~no_adsb).sum())}, frost +30 s {int(m49.sum())}, thunderstorm +200 s {int(m50.sum())}")
    if a.check:
        ref = pd.read_parquet(a.check)
        same = int((ref[config.TARGET].to_numpy() == out[config.TARGET].to_numpy()).sum())
        print(f"CHECK vs {Path(a.check).name}: identical rows {same} / {len(ref)}; bytes identical {filecmp.cmp(a.check, a.out, shallow=False)}")


if __name__ == "__main__":
    main()
