#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
python final_pipeline.py --movements tests_newdata/movements_3days.parquet --points-dir tests_newdata/points --fetch --out tests_newdata/pred_from_scratch.parquet > tests_newdata/from_scratch.log 2>&1
python - >> tests_newdata/from_scratch.log 2>&1 <<'PY'
import pandas as pd, numpy as np, sys
sys.path.insert(0, ".")
from common import io as cio
a = pd.read_parquet("tests_newdata/pred_existing_points.parquet"); b = pd.read_parquet("tests_newdata/pred_from_scratch.parquet")
base = pd.read_parquet("tests_newdata/base_3days.parquet")
m = a.merge(b, on="MVT_ID_mvt", suffixes=("_a", "_b"))
print(f"RESULT from-scratch vs existing-points: identical rows {(m.TAXITIME_SEC_mvt_a == m.TAXITIME_SEC_mvt_b).sum()} / {len(m)}")
mb = base.merge(b, on="MVT_ID_mvt", suffixes=("_base", "_b"))
print(f"RESULT base from scratch (predict_new) vs pipeline base file: same ids {len(mb)}")
y = cio.load_training(columns=["MVT_ID_mvt", "TAXITIME_SEC_mvt"]).drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
yy = pd.to_numeric(y.reindex(b.MVT_ID_mvt), errors="coerce").to_numpy(); ok = np.isfinite(yy)
r = lambda p: float(np.sqrt(np.mean((yy[ok] - p[ok]) ** 2)))
print(f"RESULT (in-sample for the 2025-trained base; functional check only) RMSE base {r(base.TAXITIME_SEC_mvt.to_numpy()):.1f} -> pipeline {r(b.TAXITIME_SEC_mvt.to_numpy()):.1f}")
PY
echo "ALL DONE" >> tests_newdata/from_scratch.log
