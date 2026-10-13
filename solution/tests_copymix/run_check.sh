#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
python tests_copymix/lfpg_check.py > tests_copymix/lfpg_check.log 2>&1
echo "EXIT $?" >> tests_copymix/lfpg_check.log
