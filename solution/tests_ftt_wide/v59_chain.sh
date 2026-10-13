#!/bin/sh
# v59 (PROBE.md "v59 probe"): wide FTT seeds 45-47 + CatBoost full seeds 45-47, then solve/build/pipeline. Log: v59_chain.log
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
L=tests_ftt_wide/v59_chain.log
echo "[$(date)] wide FTT forward_inner seeds 45-47" >> $L
python -u tests_ftt/run_ftt.py 45 46 47 --lr 5e-4 --d-token 256 --n-blocks 2 --amp --tag d256b2 --split forward_inner --epochs 260 > tests_ftt_wide/run_fwdinner_v59.log 2>&1 || { echo "[$(date)] FTT FORWARD_INNER FAILED" >> $L; exit 1; }
KEEP=$(python - <<'PY'
import numpy as np
keep = []
for s in (45, 46, 47):
    z = np.load(f"tests_ftt/preds_seed{s}_forward_inner_d256b2.npz")
    print(f"seed {s} best_epoch {int(z['best_epoch'])} hit_cap {bool(z['hit_cap'])} val {float(z['val']):.1f}", file=open("tests_ftt_wide/v59_chain.log", "a"))
    if float(z["val"]) <= 501 and not bool(z["hit_cap"]): keep.append(str(s))
print(",".join(keep))
PY
)
echo "[$(date)] wide seeds kept: $KEEP" >> $L
[ -f tests_ftt_wide/STOP ] && exit 0
if [ -n "$KEEP" ]; then
  echo "[$(date)] wide FTT full seeds $KEEP at 99 epochs" >> $L
  python -u tests_ftt/run_ftt.py $(echo $KEEP | tr ',' ' ') --lr 5e-4 --d-token 256 --n-blocks 2 --amp --tag d256b2 --split full --freeze-epochs 99 --predict-ranking > tests_ftt_wide/run_full_v59.log 2>&1 || { echo "[$(date)] FTT FULL FAILED" >> $L; exit 1; }
fi
for s in 45 46 47; do
  [ -f tests_ftt_wide/STOP ] && exit 0
  echo "[$(date)] CatBoost full seed $s" >> $L
  python pipeline_catboost_antigravity/train.py --split full --featureset with_aobt --outlier-policy keep_all --seed $s --tag cb1 \
    --freeze-from artifacts/runs/catboost_antigravity__forward-with_aobt-keep_all__seed${s}__tag=cb1/model.json > tests_ftt_wide/cb_full_$s.log 2>&1 || { echo "[$(date)] CATBOOST $s FAILED" >> $L; tail -3 tests_ftt_wide/cb_full_$s.log >> $L; exit 1; }
done
echo "[$(date)] solve" >> $L
WIDE_SEEDS="42,43,44${KEEP:+,$KEEP}" CAT_SEEDS="42,43,44,45,46,47" OUT_SUFFIX=v59 python tests_ftt_wide/solve_v58.py > tests_ftt_wide/solve_v59.log 2>&1 || { echo "[$(date)] SOLVE FAILED" >> $L; tail -5 tests_ftt_wide/solve_v59.log >> $L; exit 1; }
grep -v -i warn tests_ftt_wide/solve_v59.log >> $L
echo "[$(date)] build base" >> $L
python tests_ftt_wide/build_v58_base.py --weights tests_ftt_wide/final_weights_v59_s4.json --out tests_ftt_wide/v59_base.parquet >> $L 2>&1
python final_pipeline.py --base tests_ftt_wide/v59_base.parquet --out tests_ftt_wide/v59_candidate.parquet 2>&1 | grep -v -i warn >> $L
python - >> $L 2>&1 <<'PY'
import pandas as pd, numpy as np
a = pd.read_parquet("tests_ftt_wide/v59_candidate.parquet"); b = pd.read_parquet("submissions/fearless_barn_v58.parquet")
d = a.TAXITIME_SEC_mvt.to_numpy() - b.TAXITIME_SEC_mvt.to_numpy()
print(f"v59 vs v58: rows changed {(d != 0).sum()}, RMS change {np.sqrt(np.mean(d ** 2)):.2f} s")
PY
echo "[$(date)] ALL DONE (not uploaded)" >> $L
