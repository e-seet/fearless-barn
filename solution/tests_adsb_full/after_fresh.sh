#!/bin/sh
# Addendum 4 chain: fetch the 44 fresh 2025 days -> confirm.py -> if PASS, fetch the remaining Jan/Jul 2026 days.
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
L=tests_adsb_full/after_fresh.log
echo "[$(date)] fetching fresh 2025 days" >> $L
xargs -P 3 -I{} sh -c 'python tests_adsb_full/fetch.py {} > tests_adsb_full/fetch_{}.log 2>&1' < tests_adsb_full/fresh_days.txt
for d in $(cat tests_adsb_full/fresh_days.txt); do [ -f tests_adsb_full/points/$d.parquet ] || echo "[$(date)] MISSING $d" >> $L; done
[ -f tests_adsb_full/STOP ] && exit 0
echo "[$(date)] confirm" >> $L
python tests_adsb_full/confirm.py > tests_adsb_full/confirm.txt 2>&1
grep -E "^==|per |CONFIRMATION|Error" tests_adsb_full/confirm.txt >> $L
if grep -q "CONFIRMATION (addendum 4): PASS" tests_adsb_full/confirm.txt && [ ! -f tests_adsb_full/STOP ]; then
  echo "[$(date)] PASS -> fetching remaining 2026 Jan/Jul days" >> $L
  python -c "
import pandas as pd; from pathlib import Path
P=Path('tests_adsb_full/points')
print('\n'.join(d.strftime('%Y-%m-%d') for m in ('2026-01','2026-07') for d in pd.date_range(m+'-01',periods=31) if not (P/(d.strftime('%Y-%m-%d')+'.parquet')).exists()))" > tests_adsb_full/days_2026.txt
  xargs -P 3 -I{} sh -c 'python tests_adsb_full/fetch.py {} > tests_adsb_full/fetch_{}.log 2>&1' < tests_adsb_full/days_2026.txt
  for d in $(cat tests_adsb_full/days_2026.txt); do [ -f tests_adsb_full/points/$d.parquet ] || echo "[$(date)] MISSING $d" >> $L; done
fi
echo "[$(date)] ALL DONE" >> $L
