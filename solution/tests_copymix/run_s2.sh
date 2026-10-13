#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python final_pipeline.py --base tests_ftt_wide/v59_base.parquet --check submissions/fearless_barn_v61.parquet --out tests_copymix/v61_rebuild.parquet --s2-out tests_copymix/v61_s2.parquet > tests_copymix/s2.log 2>&1
echo "EXIT $?" >> tests_copymix/s2.log
