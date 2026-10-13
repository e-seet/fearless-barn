#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
python tests_copymix/build_M_ranking.py > tests_copymix/build_M.log 2>&1
echo "EXIT $?" >> tests_copymix/build_M.log
