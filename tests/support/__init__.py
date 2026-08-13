"""Shared test harness. Imported as `tests.support.*`, never collected.

`tests/` has no `__init__.py` on purpose -- adding one would change pytest's
basedir computation for all 38 existing test files. This package resolves
instead through `pythonpath = ["."]` in pyproject.toml plus PEP 420 namespace
packages, which is also what makes `bench` importable under bare `pytest`.
"""
