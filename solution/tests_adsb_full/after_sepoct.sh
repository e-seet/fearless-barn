#!/bin/sh
# Smoke-test evaluate_v45.py once 2 Sep-Oct days exist, then the real run once all 61 exist (or 9 h).
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
L=tests_adsb_full/after_sepoct.log
python -c "
import time; from pathlib import Path
while len(list(Path('tests_adsb_full/points').glob('2025-09-*.parquet')) + list(Path('tests_adsb_full/points').glob('2025-10-*.parquet'))) < 2: time.sleep(60)"
echo "[$(date)] SMOKE (2 days, not the result)" >> $L
python tests_adsb_full/evaluate_v45.py 2>&1 | grep -v -i warning | tail -15 >> $L; rm -f tests_adsb_full/evaluate_v45.json
python - <<'PY'
import time
from pathlib import Path
days = Path("tests_adsb_full/sepoct_days.txt").read_text().split(); t0 = time.time()
while time.time() - t0 < 9 * 3600 and not all(Path(f"tests_adsb_full/points/{d}.parquet").exists() for d in days):
    time.sleep(120)
PY
for d in $(cat tests_adsb_full/sepoct_days.txt); do [ -f tests_adsb_full/points/$d.parquet ] || echo "[$(date)] MISSING $d" >> $L; done
[ -f tests_adsb_full/STOP ] && exit 0
echo "[$(date)] REAL RUN" >> $L
python tests_adsb_full/evaluate_v45.py > tests_adsb_full/evaluate_v45.txt 2>&1
grep -v -i warning tests_adsb_full/evaluate_v45.txt >> $L
echo "[$(date)] ALL DONE" >> $L
