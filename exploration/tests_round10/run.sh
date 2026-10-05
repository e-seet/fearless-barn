#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python tests_round10/round10.py > tests_round10/round10.log 2>&1
echo "ALL DONE" >> tests_round10/round10.log
