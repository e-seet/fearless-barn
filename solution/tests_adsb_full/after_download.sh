#!/bin/sh
# Runs once the 2025 staged fetch has finished (all 18 Jan/Jul 2025 parquet files present and no
# fetch.py running): the pre-registered evaluation (3 variants), then the 2026 days the drift gate
# needs, then the drift report. Everything logs to after_download.log. STOP file aborts between steps.
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
L=tests_adsb_full/after_download.log
DAYS25="2025-01-06 2025-01-09 2025-01-13 2025-01-15 2025-01-18 2025-01-21 2025-01-24 2025-01-28 2025-01-31 2025-07-03 2025-07-07 2025-07-10 2025-07-15 2025-07-17 2025-07-20 2025-07-23 2025-07-26 2025-07-30"
echo "[$(date)] waiting for the 2025 fetch" >> $L
python - <<'EOF'
import time
from pathlib import Path
P = Path("tests_adsb_full/points")
days = "2025-01-06 2025-01-09 2025-01-13 2025-01-15 2025-01-18 2025-01-21 2025-01-24 2025-01-28 2025-01-31 2025-07-03 2025-07-07 2025-07-10 2025-07-15 2025-07-17 2025-07-20 2025-07-23 2025-07-26 2025-07-30".split()
t0 = time.time()
while True:
    have = all((P / f"{d}.parquet").exists() for d in days)
    if have or time.time() - t0 > 5 * 3600:   # a day's parquet is written only when that day is complete;
        break                                  # after 5 h go on with what exists (missing days are logged)
    time.sleep(120)
EOF
for d in $DAYS25; do [ -f tests_adsb_full/points/$d.parquet ] || echo "[$(date)] MISSING $d" >> $L; done
for v in main move all; do
  [ -f tests_adsb_full/STOP ] && exit 0
  echo "[$(date)] evaluate $v" >> $L
  python tests_adsb_full/evaluate.py $v > tests_adsb_full/evaluate_$v.txt 2>&1
  tail -n 12 tests_adsb_full/evaluate_$v.txt >> $L
done
[ -f tests_adsb_full/STOP ] && exit 0
echo "[$(date)] fetching 2026 drift days" >> $L
printf '%s\n' 2026-01-20 2026-01-27 2026-07-08 2026-07-15 2026-07-22 | \
  xargs -P 3 -I{} sh -c 'python tests_adsb_full/fetch.py {} > tests_adsb_full/fetch_{}.log 2>&1'
echo "[$(date)] drift report" >> $L
python tests_adsb_full/drift.py > tests_adsb_full/drift.txt 2>&1
cat tests_adsb_full/drift.txt >> $L
echo "[$(date)] ALL DONE" >> $L
