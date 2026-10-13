#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python tests_copymix/copymix.py > tests_copymix/copymix.log 2>&1
echo "EXIT $?" >> tests_copymix/copymix.log
