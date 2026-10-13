#!/bin/sh
# After the re-download + re-confirmation: drift on the complete 2026 set, then the final build.
# Stops before any upload (user decision). Log: after_redo.log
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
L=tests_adsb_full/after_redo.log
echo "[$(date)] waiting for re-download and re-confirmation" >> $L
python - <<'PY'
import time
from pathlib import Path
f = Path("tests_adsb_full/confirm_after_redo.txt"); t0 = time.time()
while time.time() - t0 < 10 * 3600:
    if f.exists() and ("CONFIRMATION" in f.read_text(errors="ignore") or "Error" in f.read_text(errors="ignore")):
        break
    time.sleep(120)
PY
grep -E "^==|CONFIRMATION|Error" tests_adsb_full/confirm_after_redo.txt >> $L
for d in $(cat tests_adsb_full/redo_days.txt); do [ -f tests_adsb_full/points/$d.parquet ] || echo "[$(date)] STILL MISSING $d" >> $L; done
if ! grep -q "CONFIRMATION (addendum 4): PASS" tests_adsb_full/confirm_after_redo.txt; then
  echo "[$(date)] confirmation did not PASS after the re-download -> no build" >> $L; exit 0
fi
[ -f tests_adsb_full/STOP ] && exit 0
echo "[$(date)] drift (complete set)" >> $L
python tests_adsb_full/drift.py > tests_adsb_full/drift_full.txt 2>&1; grep -E "==|DRIFT" tests_adsb_full/drift_full.txt >> $L
echo "[$(date)] final build" >> $L
python tests_adsb_full/final_build.py > tests_adsb_full/final_build.log 2>&1; cat tests_adsb_full/build_report.txt >> $L 2>/dev/null || tail -5 tests_adsb_full/final_build.log >> $L
echo "[$(date)] ALL DONE (nothing uploaded)" >> $L
