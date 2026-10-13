#!/bin/sh
# PREREGISTERED_tree.md chain: wait for Feb-Jun fetch -> retry missing days once -> 3 adsb tree runs ->
# dump predictions -> decision -> (PASS) full refit. Log: after_febjun.log. STOP file aborts between steps.
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
L=tests_adsb_full/after_febjun.log
echo "[$(date)] waiting for the Feb-Jun fetch" >> $L
python - <<'PY'
import time, subprocess
from pathlib import Path
days = Path("tests_adsb_full/febjun_days.txt").read_text().split()
def fetching():
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'fetch.py 2025-0[2-6]' }).Count"],
        capture_output=True, text=True)
    return (r.stdout.strip() or "0") != "0"
t0 = time.time()
while time.time() - t0 < 30 * 3600:
    if all(Path(f"tests_adsb_full/points/{d}.parquet").exists() for d in days):
        break
    if not fetching():
        time.sleep(300)
        if not fetching():
            break
    time.sleep(120)
PY
for d in $(cat tests_adsb_full/febjun_days.txt); do
  if [ ! -f tests_adsb_full/points/$d.parquet ]; then
    echo "[$(date)] retry $d" >> $L
    python tests_adsb_full/fetch.py $d > tests_adsb_full/fetch_$d.log 2>&1
    [ -f tests_adsb_full/points/$d.parquet ] || echo "[$(date)] STILL MISSING $d" >> $L
  fi
done
python -c "
import json,glob
b=[json.load(open(f)) for f in glob.glob('tests_adsb_full/points/2025-0[2-6]-*.json')]
print('Feb-Jun days:', len(b), '| suspect:', [r['day'] for r in b if r['gb'] < 1.5 or r['files'] < 30000])" >> $L
for s in forward_inner forward seasonal_full; do
  [ -f tests_adsb_full/STOP ] && exit 0
  echo "[$(date)] train adsb tree: $s" >> $L
  python pipeline_airport_operator_blend/train.py --split $s --featureset with_aobt --outlier-policy keep_all --seed 42 --adsb --tag adsb > tests_adsb_full/train_tree_$s.log 2>&1 || { echo "[$(date)] TRAIN FAILED $s" >> $L; tail -5 tests_adsb_full/train_tree_$s.log >> $L; exit 1; }
  grep -E "alpha|val RMSE" tests_adsb_full/train_tree_$s.log >> $L
done
echo "[$(date)] dump adsb tree predictions" >> $L
python tests_adsb_full/dump_tree_preds.py adsb 2>&1 | grep -v -i warn >> $L
echo "[$(date)] decision" >> $L
python tests_adsb_full/evaluate_tree.py > tests_adsb_full/evaluate_tree.txt 2>&1
grep -v -i warn tests_adsb_full/evaluate_tree.txt >> $L
if grep -q "=> PASS" tests_adsb_full/evaluate_tree.txt && [ ! -f tests_adsb_full/STOP ]; then
  echo "[$(date)] PASS -> full refit (frozen from forward adsb)" >> $L
  python pipeline_airport_operator_blend/train.py --split full --featureset with_aobt --outlier-policy keep_all --seed 42 --adsb --tag adsb \
    --freeze-from artifacts/runs/airport_operator_blend__forward-with_aobt-keep_all__seed42__tag=adsb/model.json > tests_adsb_full/train_tree_full.log 2>&1 \
    && echo "[$(date)] full refit done" >> $L || { echo "[$(date)] FULL REFIT FAILED" >> $L; tail -5 tests_adsb_full/train_tree_full.log >> $L; }
fi
echo "[$(date)] ALL DONE (nothing built or uploaded)" >> $L
