#!/usr/bin/env bash
set -euo pipefail
D=/data/orca/workspaces/fmls-numpy-E-fullsuite-20260928
trap 'printf "%s\n" "$?" > "$D/non-f2py-exit.status"' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
nice -n 19 "$D/.venv/bin/python" -m pytest -q --disable-warnings -x --ignore="$D/.venv/lib/python3.12/site-packages/numpy/f2py/tests" "$D/.venv/lib/python3.12/site-packages/numpy" > "$D/non-f2py-tests.log" 2>&1
