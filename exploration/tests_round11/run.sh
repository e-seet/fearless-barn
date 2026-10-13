#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python tests_round11/round11.py > tests_round11/round11.log 2>&1
echo "EXIT $?" >> tests_round11/round11.log
