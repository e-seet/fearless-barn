#!/usr/bin/env python3
"""v56 = v55 + LFPG ADS-B corrector (round 9 test C PASS). Trained on all 7 held-out months' LFPG rows with a strict
track and an NM record (v48 design, single-airport inputs, seeds 42-46); applied to 2026 LFPG departures with a track
and a record; the corrector's `base` input is the v42 value, the correction is added to v55. Every other row is
byte-identical to v55. Saves model_v56/, reload check. Does not upload."""
import json, sys, filecmp
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_adsb_full"))
import round9 as R
import features
from common import io as cio

df = R.frame()
trc = df[R.lfpg(df)]
MD = HERE / "model_v56"; MD.mkdir(exist_ok=True)
models = []
for s in R.SEEDS:
    m = lgb.LGBMRegressor(**{**R.PARAMS, "random_state": s}).fit(trc[R.FEAT_1], trc.y - trc.base)
    m.booster_.save_model(str(MD / f"lfpg_corrector_seed{s}.txt")); models.append(m)
(MD / "config.json").write_text(json.dumps({"airport": "LFPG", "features": R.FEAT_1, "adsb_features": R.FEAT_ADSB, "params": R.PARAMS,
    "seeds": R.SEEDS, "n_train": int(len(trc)), "train_months": sorted(trc.mon.unique().tolist()),
    "applies_to": "2026 LFPG departures with FLIGHT_ID_mvt and a strict ADS-B track; added to v55"}, indent=1))

SUB = ROOT / "submissions"
v55 = pd.read_parquet(SUB / "fearless_barn_v55.parquet"); v42 = pd.read_parquet(SUB / "fearless_barn_v42.parquet")
assert v55.MVT_ID_mvt.equals(v42.MVT_ID_mvt)
rk = cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"])
rk = rk[rk.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt"); rk = rk.set_index(rk.MVT_ID_mvt.astype("int64"))
ids = v55.MVT_ID_mvt.astype("int64").to_numpy()
te = pd.DataFrame({"base": v42.TAXITIME_SEC_mvt.astype(float).to_numpy()}, index=ids)
t = pd.to_datetime(rk.MVT_TIME_UTC_mvt.reindex(ids), utc=True)
te["ap"] = rk.ADEP_mvt.reindex(ids).astype(str).to_numpy(); te["hour"] = t.dt.hour.to_numpy()
te["rec"] = rk.FLIGHT_ID_mvt.reindex(ids).notna().to_numpy()
days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("2026-0[17]-*.parquet"))
e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]; e.index = e.index.astype("int64")
te = te.join(e[R.FEAT_ADSB])
c = (te.ap.eq("LFPG") & te.rec & te.e_first.notna()).to_numpy()


def build(fn):
    corr = np.zeros(len(te)); corr[c] = fn(te.loc[c, R.FEAT_1])
    out = v55.copy()
    out["TAXITIME_SEC_mvt"] = np.where(c, np.clip(np.rint(v55.TAXITIME_SEC_mvt.to_numpy() + corr), 60, 86400),
                                       v55.TAXITIME_SEC_mvt.to_numpy()).astype("int32")
    return corr, out


corr, out = build(lambda X: np.mean([m.predict(X) for m in models], axis=0))
out.to_parquet(HERE / "v56_candidate.parquet", index=False)
bs = [lgb.Booster(model_file=str(MD / f"lfpg_corrector_seed{s}.txt")) for s in R.SEEDS]
_, out2 = build(lambda X: np.mean([b.predict(X) for b in bs], axis=0))
out2.to_parquet(HERE / "v56_rebuilt.parquet", index=False)
lfpg_rows = te.ap.eq("LFPG").to_numpy(); mon = t.dt.month.to_numpy()
changed = (out.TAXITIME_SEC_mvt != v55.TAXITIME_SEC_mvt).to_numpy()
lines = [f"training: {len(trc)} LFPG rows (track + record), months {sorted(trc.mon.unique().tolist())}",
         f"2026 LFPG rows {int(lfpg_rows.sum())}; corrected {int(c.sum())} ({c.sum() / lfpg_rows.sum():.1%}): Jan {int((c & (mon == 1)).sum())}, Jul {int((c & (mon == 7)).sum())}",
         f"correction mean {corr[c].mean():+.1f} s, mean |.| {np.abs(corr[c]).mean():.1f}, 1-99% [{np.percentile(corr[c], 1):.0f}, {np.percentile(corr[c], 99):.0f}]",
         f"rows changed vs v55: {int(changed.sum())}; all changed rows are LFPG: {bool((changed <= c).all())}",
         f"reload-from-saved-model rebuild byte-identical: {filecmp.cmp(HERE / 'v56_candidate.parquet', HERE / 'v56_rebuilt.parquet', shallow=False)}"]
print("\n".join(lines))
(HERE / "build_report_v56.txt").write_text("\n".join(lines) + "\n")
