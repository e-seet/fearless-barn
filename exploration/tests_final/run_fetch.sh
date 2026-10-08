#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
cat tests_final/days_missing.txt | xargs -P 4 -I{} sh -c 'python tests_adsb_full/fetch.py {} --movements ../prc-2026-datasets/final_ranking.parquet > tests_final/fetch_{}.log 2>&1; echo "{} EXIT $?" >> tests_final/fetch_done.txt'
echo ALLFETCHED >> tests_final/fetch_done.txt
