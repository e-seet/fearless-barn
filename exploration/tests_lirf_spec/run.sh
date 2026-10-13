#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python tests_lirf_spec/spec.py > tests_lirf_spec/spec.log 2>&1
echo "EXIT $?" >> tests_lirf_spec/spec.log
