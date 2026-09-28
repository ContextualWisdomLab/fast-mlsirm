#!/usr/bin/env bash
set -euo pipefail
D=/data/orca/workspaces/fmls-numpy-E-fullsuite-20260928
UV=/home/seongho/.local/bin/uv
trap 'printf "%s\n" "$?" > "$D/tests-exit.status"' EXIT
"$UV" pip install --offline --python "$D/.venv/bin/python" pytest==9.1.1 hypothesis==6.168.0 > "$D/install-tests-cached.log" 2>&1
"$D/.venv/bin/python" -c 'import numpy, pytest, hypothesis; print(numpy.__version__, pytest.__version__, hypothesis.__version__, numpy.__file__)' > "$D/versions.log"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
nice -n 19 "$D/.venv/bin/python" -m pytest -q --disable-warnings -x "$D/.venv/lib/python3.12/site-packages/numpy" > "$D/full-tests.log" 2>&1
