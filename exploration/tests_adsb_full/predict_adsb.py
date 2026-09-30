#!/usr/bin/env python3
"""Apply the saved ADS-B corrector (model/adsb_corrector.txt, model/config.json) on top of a base
submission file. Inputs: the base file (default v42) and per-day ADS-B points (fetch.py) for the days
of the rows to predict. Output: a submission file in the same format.

usage: python tests_adsb_full/predict_adsb.py [--base submissions/fearless_barn_v42.parquet] [--out FILE] [--check FILE]
  --check compares the output with an existing file byte for byte (reproducibility of v43)."""
import argparse, json, sys, filecmp
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import features
from common import io as cio

ap_ = argparse.ArgumentParser()
ap_.add_argument("--base", default=str(ROOT / "submissions" / "fearless_barn_v42.parquet"))
ap_.add_argument("--out", default=str(HERE / "v43_rebuilt.parquet"))
ap_.add_argument("--check", default=None)
a = ap_.parse_args()
cfg = json.loads((HERE / "model" / "config.json").read_text())
booster = lgb.Booster(model_file=str(HERE / "model" / "adsb_corrector.txt"))
SHIP, FEAT, FEAT_ADSB = cfg["airports"], cfg["features"], cfg["adsb_features"]

base = pd.read_parquet(a.base)
rk = cio.load_ranking(columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
rk.index = rk.index.astype("int64")
te = pd.DataFrame({"base": base.TAXITIME_SEC_mvt.astype(float).to_numpy()}, index=base.MVT_ID_mvt.astype("int64"))
te["ap"] = rk.ADEP_mvt.reindex(te.index).astype(str).to_numpy()
tt = pd.to_datetime(rk.MVT_TIME_UTC_mvt.reindex(te.index), utc=True)
te["hour"] = tt.dt.hour.to_numpy()
days = sorted(set(tt.dt.strftime("%Y-%m-%d").dropna()) | set((tt + pd.Timedelta(hours=4)).dt.strftime("%Y-%m-%d").dropna()))
est = [features.load_day(d) for d in days if (HERE / "points" / f"{d}.parquet").exists()]
e = pd.concat(est).sort_values("liftoff_dt"); e = e[~e.index.duplicated()]; e.index = e.index.astype("int64")
te = te.join(e[FEAT_ADSB])
te["covered"] = te.e_first.notna() & te.ap.isin(SHIP)
te["apc"] = pd.Categorical(te.ap.where(te.ap.isin(SHIP)), categories=SHIP)
c = te.covered.to_numpy()
corr = np.zeros(len(te)); corr[c] = booster.predict(te.loc[c, FEAT])
out = base.copy(); out["TAXITIME_SEC_mvt"] = np.clip(np.rint(te.base.to_numpy() + corr), 0, 86400).astype("int32")
out.to_parquet(a.out, index=False)
print(f"wrote {a.out}: {len(out)} rows, {int(c.sum())} corrected")
if a.check:
    ref = pd.read_parquet(a.check)
    same = int((ref.TAXITIME_SEC_mvt.to_numpy() == out.TAXITIME_SEC_mvt.to_numpy()).sum())
    print(f"CHECK vs {Path(a.check).name}: identical rows {same} / {len(ref)}; ids equal {ref.MVT_ID_mvt.equals(out.MVT_ID_mvt)}; "
          f"bytes identical {filecmp.cmp(a.check, a.out, shallow=False)}")
