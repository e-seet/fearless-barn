#!/bin/sh
# After the wide-FTT queue: solve v58 weights, pick S3/S4 (Amendment 1), build the base, run final_pipeline -> v58 candidate.
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
L=tests_ftt_wide/build_chain.log
until grep -qE "ALL DONE|FAILED" tests_ftt_wide/queue.log; do sleep 60; done
grep -q "ALL DONE" tests_ftt_wide/queue.log || { echo "[$(date)] training failed; no build" >> $L; exit 1; }
echo "[$(date)] solve" >> $L
python tests_ftt_wide/solve_v58.py > tests_ftt_wide/solve_v58.log 2>&1 || { echo "[$(date)] SOLVE FAILED" >> $L; tail -5 tests_ftt_wide/solve_v58.log >> $L; exit 1; }
grep -v -i warn tests_ftt_wide/solve_v58.log >> $L
if grep -q "NO airport" tests_ftt_wide/solve_v58.log; then S=s3; else S=s4; fi
echo "[$(date)] build base $S" >> $L
python tests_ftt_wide/build_v58_base.py --weights tests_ftt_wide/final_weights_v58_$S.json --out tests_ftt_wide/v58_base.parquet >> $L 2>&1
python tests_ftt_wide/build_v58_base.py --weights tests_ftt_wide/final_weights_v58_s3.json --out tests_ftt_wide/v58_base_s3.parquet >> $L 2>&1
echo "[$(date)] final pipeline on the new base ($S)" >> $L
python final_pipeline.py --base tests_ftt_wide/v58_base.parquet --out tests_ftt_wide/v58_candidate.parquet 2>&1 | grep -v -i warn >> $L
python - >> $L 2>&1 <<'PY'
import pandas as pd, numpy as np
a = pd.read_parquet("tests_ftt_wide/v58_candidate.parquet"); b = pd.read_parquet("submissions/fearless_barn_v57.parquet")
base_new = pd.read_parquet("tests_ftt_wide/v58_base.parquet"); base_old = pd.read_parquet("submissions/fearless_barn_v42.parquet")
d = a.TAXITIME_SEC_mvt.to_numpy() - b.TAXITIME_SEC_mvt.to_numpy(); db = base_new.TAXITIME_SEC_mvt.to_numpy() - base_old.TAXITIME_SEC_mvt.to_numpy()
print(f"v58 vs v57: rows changed {(d != 0).sum()}, RMS change {np.sqrt(np.mean(d**2)):.2f} s | base v58 vs v42: RMS change {np.sqrt(np.mean(db**2)):.2f} s")
PY
echo "[$(date)] ALL DONE (not uploaded)" >> $L
