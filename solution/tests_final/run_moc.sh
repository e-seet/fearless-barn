#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/solution || exit 1
python tests_final/months_of_concern.py > tests_final/months_of_concern.log 2>&1
echo "EXIT $?" >> tests_final/months_of_concern.log
