#!/bin/sh
# Waits for the Nov-Dec 2025 fetch (all 61 days, or 9 h), then runs the preregistered extension test.
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
L=tests_adsb_full/after_novdec.log
echo "[$(date)] waiting for Nov-Dec fetch" >> $L
python - <<'PY'
import time
from pathlib import Path
days = Path("tests_adsb_full/novdec_days.txt").read_text().split(); t0 = time.time()
while time.time() - t0 < 9 * 3600 and not all(Path(f"tests_adsb_full/points/{d}.parquet").exists() for d in days):
    time.sleep(120)
PY
for d in $(cat tests_adsb_full/novdec_days.txt); do [ -f tests_adsb_full/points/$d.parquet ] || echo "[$(date)] MISSING $d" >> $L; done
grep -l INCOMPLETE tests_adsb_full/fetch_2025-1[12]-*.log >> $L 2>/dev/null
[ -f tests_adsb_full/STOP ] && exit 0
echo "[$(date)] evaluate_ext" >> $L
python tests_adsb_full/evaluate_ext.py > tests_adsb_full/evaluate_ext.txt 2>&1
grep -v -i warning tests_adsb_full/evaluate_ext.txt >> $L
echo "[$(date)] ALL DONE" >> $L
