#!/bin/sh
# After round 10 finishes: wide-shallow FTT, forward_inner then full (tests_ftt_wide/PROBE.md).
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
L=tests_ftt_wide/queue.log
echo "[$(date)] waiting for round 10" >> $L
until grep -q "ALL DONE" tests_round10/round10.log 2>/dev/null; do sleep 60; done
[ -f tests_ftt_wide/STOP ] && exit 0
echo "[$(date)] forward_inner seeds 42-44" >> $L
python -u tests_ftt/run_ftt.py 42 43 44 --lr 5e-4 --d-token 256 --n-blocks 2 --amp --tag d256b2 --split forward_inner --epochs 260 > tests_ftt_wide/run_fwdinner.log 2>&1 || { echo "[$(date)] FORWARD_INNER FAILED" >> $L; exit 1; }
python - >> $L 2>&1 <<'PY'
import numpy as np
for s in (42, 43, 44):
    z = np.load(f"tests_ftt/preds_seed{s}_forward_inner_d256b2.npz")
    print("seed", s, "best_epoch", int(z["best_epoch"]), "epochs_run", int(z["epochs_run"]), "hit_cap", bool(z["hit_cap"]), "val", round(float(z["val"]), 2))
PY
[ -f tests_ftt_wide/STOP ] && exit 0
E=$(python -c "import math, numpy as np; print(math.ceil(float(np.median([int(np.load(f'tests_ftt/preds_seed{s}_forward_inner_d256b2.npz')['best_epoch']) for s in (42,43,44)]))))")
echo "[$(date)] full seeds 42-44 at $E epochs" >> $L
python -u tests_ftt/run_ftt.py 42 43 44 --lr 5e-4 --d-token 256 --n-blocks 2 --amp --tag d256b2 --split full --freeze-epochs $E --predict-ranking > tests_ftt_wide/run_full.log 2>&1 || { echo "[$(date)] FULL FAILED" >> $L; exit 1; }
echo "[$(date)] ALL DONE" >> $L
