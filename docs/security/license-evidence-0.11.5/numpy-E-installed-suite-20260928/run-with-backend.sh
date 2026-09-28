#!/usr/bin/env bash
set -euo pipefail
D=/data/orca/workspaces/fmls-numpy-E-fullsuite-20260928
trap 'printf "%s\n" "$?" > "$D/with-backend-exit.status"' EXIT
export PATH="$D/.venv/bin:$PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
nice -n 19 "$D/.venv/bin/python" -m pytest -q --disable-warnings -x "$D/.venv/lib/python3.12/site-packages/numpy" > "$D/with-backend-tests.log" 2>&1
