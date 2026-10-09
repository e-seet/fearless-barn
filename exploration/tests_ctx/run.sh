#!/bin/sh
cd /c/Users/Admin/Desktop/eddie/fearless-barn/exploration || exit 1
python tests_ctx/ctx.py > tests_ctx/ctx.log 2>&1
echo "EXIT $?" >> tests_ctx/ctx.log
