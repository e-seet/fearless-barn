#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
python predict_new.py --weights tests_ftt_wide/final_weights_v59_s4.json --ids ../prc-2026-datasets/submitting.parquet --check tests_ftt_wide/v59_base.parquet --out tests_final/v59_base_rebuilt.parquet > tests_final/base_check.log 2>&1
echo "EXIT $?" >> tests_final/base_check.log
