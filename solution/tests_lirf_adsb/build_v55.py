#!/usr/bin/env python3
"""v55 = v50 + LIRF ADS-B corrector (PREREGISTERED.md PASS), on January 2026 LIRF departures with an NM record and a
strict-matched ADS-B track. Corrector trained on all Jul-Dec 2025 eligible LIRF rows (held-out bases), 5 seeds.
The corrector's `base` input is the v42 value (the model base it was trained against); the correction is added to v50.
Saves model_v55/, writes v55_candidate.parquet, reloads the saved boosters and checks byte identity. Does not upload."""
import json, sys, filecmp
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; exec(open(ROOT.parent / "tools" / "repo_layout.py").read()); ROOT = dual(ROOT)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests_adsb_full"))
src = (HERE / "lirf_lomo.py").read_text(encoding="utf-8").split("def shuffled")[0]
exec(compile(src, "lirf_lomo_frame", "exec"))          # builds `df` (Jul-Dec 2025 LIRF rows), FEAT, PARAMS, SEEDS, features
import features
from common import io as cio

MD = HERE / "model_v55"; MD.mkdir(exist_ok=True)
trc = df[df.elig]
models = []
for s in SEEDS:
    m = lgb.LGBMRegressor(**{**PARAMS, "random_state": s}).fit(trc[FEAT], trc.y - trc.base)
    m.booster_.save_model(str(MD / f"lirf_corrector_seed{s}.txt")); models.append(m)
(MD / "config.json").write_text(json.dumps({"airport": "LIRF", "features": FEAT, "adsb_features": FEAT_ADSB, "params": PARAMS,
    "seeds": SEEDS, "train_months": sorted(df.mon.unique().tolist()), "n_train": int(len(trc)), "lirf_tabnet_weight": TW,
    "applies_to": "Jan 2026 LIRF departures with FLIGHT_ID_mvt and a strict ADS-B track; added to v50"}, indent=1))

SUB = ROOT / "submissions"
v50 = pd.read_parquet(SUB / "fearless_barn_v50.parquet"); v42 = pd.read_parquet(SUB / "fearless_barn_v42.parquet")
assert v50.MVT_ID_mvt.equals(v42.MVT_ID_mvt)
rk = cio.load_ranking(columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_ID_mvt"])
rk = rk[rk.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index(rk[rk.PHASE_mvt.eq("DEP")].drop_duplicates("MVT_ID_mvt").MVT_ID_mvt.astype("int64"))
ids = v50.MVT_ID_mvt.astype("int64").to_numpy()
te = pd.DataFrame({"base": v42.TAXITIME_SEC_mvt.astype(float).to_numpy()}, index=ids)
t = pd.to_datetime(rk.MVT_TIME_UTC_mvt.reindex(ids), utc=True)
te["ap"] = rk.ADEP_mvt.reindex(ids).astype(str).to_numpy(); te["hour"] = t.dt.hour.to_numpy(); te["mon"] = t.dt.month.to_numpy()
te["rec"] = rk.FLIGHT_ID_mvt.reindex(ids).notna().to_numpy()
days = sorted(f.stem for f in (ROOT / "tests_adsb_full" / "points").glob("2026-01-*.parquet"))
e = pd.concat([features.load_day(d) for d in days]).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]; e.index = e.index.astype("int64")
te = te.join(e[FEAT_ADSB])
c = (te.ap.eq("LIRF") & (te.mon == 1) & te.rec & te.e_first.notna()).to_numpy()


def build(pred_fn):
    corr = np.zeros(len(te)); corr[c] = pred_fn(te.loc[c, FEAT])
    out = v50.copy()
    out["TAXITIME_SEC_mvt"] = np.where(c, np.clip(np.rint(v50.TAXITIME_SEC_mvt.to_numpy() + corr), 60, 86400),
                                       v50.TAXITIME_SEC_mvt.to_numpy()).astype("int32")
    return corr, out


corr, out = build(lambda X: np.mean([m.predict(X) for m in models], axis=0))
out.to_parquet(HERE / "v55_candidate.parquet", index=False)
bs = [lgb.Booster(model_file=str(MD / f"lirf_corrector_seed{s}.txt")) for s in SEEDS]
_, out2 = build(lambda X: np.mean([b.predict(X) for b in bs], axis=0))
out2.to_parquet(HERE / "v55_rebuilt.parquet", index=False)
jan_lirf = (te.ap.eq("LIRF") & (te.mon == 1)).to_numpy()
lines = [f"training: {len(trc)} eligible LIRF rows, months {sorted(df.mon.unique().tolist())}",
         f"Jan 2026 LIRF rows {int(jan_lirf.sum())}; corrected {int(c.sum())} ({c.sum() / jan_lirf.sum():.0%})",
         f"correction mean {corr[c].mean():+.1f} s, mean |.| {np.abs(corr[c]).mean():.1f}, 1-99% [{np.percentile(corr[c], 1):.0f}, {np.percentile(corr[c], 99):.0f}]",
         f"rows changed vs v50: {(out.TAXITIME_SEC_mvt != v50.TAXITIME_SEC_mvt).sum()}; all changed rows are Jan LIRF: {bool(((out.TAXITIME_SEC_mvt != v50.TAXITIME_SEC_mvt).to_numpy() <= c).all())}",
         f"reload-from-saved-model rebuild byte-identical: {filecmp.cmp(HERE / 'v55_candidate.parquet', HERE / 'v55_rebuilt.parquet', shallow=False)}"]
print("\n".join(lines))
(HERE / "build_report_v55.txt").write_text("\n".join(lines) + "\n")
