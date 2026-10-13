#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python tests_round13/round13.py > tests_round13/round13.log 2>&1
echo "EXIT $?" >> tests_round13/round13.log
