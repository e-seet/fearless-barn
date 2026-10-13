#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
until grep -q "^EXIT" tests_final/base_check.log; do sleep 30; done
grep -q "^EXIT 0" tests_final/base_check.log || { echo "BASECHECK FAILED" >> tests_final/final_build.log; exit 1; }
python predict_new.py --movements ../prc-2026-datasets/final_ranking.parquet --ids ../prc-2026-datasets/final_submitting.parquet --weights tests_ftt_wide/final_weights_v59_s4.json --out tests_final/final_base.parquet > tests_final/final_base.log 2>&1
echo "BASE EXIT $?" >> tests_final/final_build.log
python final_pipeline.py --movements ../prc-2026-datasets/final_ranking.parquet --base tests_final/final_base.parquet --out submissions/fearless-barn_final.parquet --s2-out tests_final/final_s2.parquet > tests_final/final_pipeline.log 2>&1
echo "PIPELINE EXIT $?" >> tests_final/final_build.log
