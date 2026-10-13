#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
python tests_round12/round12.py > tests_round12/round12.log 2>&1
echo "EXIT $?" >> tests_round12/round12.log
