#!/bin/sh
# Wait for train_chain.sh to finish (ALL DONE or a FAILED line), then run the preregistered test.
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python -c "
import time; from pathlib import Path
f = Path('tests_febjun/train_chain.log'); t0 = time.time()
while time.time() - t0 < 14 * 3600:
    s = f.read_text(errors='ignore') if f.exists() else ''
    if 'ALL DONE' in s or 'FAILED' in s: break
    time.sleep(120)"
grep -q "ALL DONE" tests_febjun/train_chain.log || { echo "[$(date)] training did not finish; test not run" >> tests_febjun/train_chain.log; exit 1; }
python tests_febjun/lomo_febjun.py > tests_febjun/lomo_febjun.txt 2>&1
grep -v -i warn tests_febjun/lomo_febjun.txt >> tests_febjun/train_chain.log
echo "[$(date)] TEST DONE" >> tests_febjun/train_chain.log
