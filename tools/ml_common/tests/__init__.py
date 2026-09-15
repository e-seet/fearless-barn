"""Tests for ml_common.

A real `__init__.py` (not an implicit namespace package) is required: MISTAKES.md #13
and #17 both record a locally-named `tests/` directory being silently shadowed by an
unrelated `tests` package in the shared interpreter's site-packages, because Python
resolves a REGULAR package found anywhere on sys.path before a namespace package at an
earlier path entry. The wrong module is then imported with no error.

Run from the workspace root:

    python3 -m unittest discover -s tools/ml_common/tests -t tools -v
"""
