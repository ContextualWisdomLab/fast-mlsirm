#!/usr/bin/env bash
set -euo pipefail
D=/data/orca/workspaces/fmls-numpy-E-fullsuite-20260928
W=/data/orca/workspaces/fmls-numpy-stable-E-validation-20260928/input/numpy-2.5.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
SHA=91c7a71387703c23a9bcf953adcc7b0d3ec8f41e6e9a090afaa3a6532c6c8604
UV=/home/seongho/.local/bin/uv
PY=/home/seongho/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
trap 'printf "%s\n" "$?" > "$D/exit.status"' EXIT
mkdir "$D/input"
cp "$W" "$D/input/"
test "$(sha256sum "$D/input/$(basename "$W")" | cut -d' ' -f1)" = "$SHA"
"$UV" venv --python "$PY" "$D/.venv" > "$D/venv.log" 2>&1
"$UV" pip install --offline --python "$D/.venv/bin/python" --no-deps "$D/input/$(basename "$W")" > "$D/install-numpy.log" 2>&1
"$UV" pip install --offline --python "$D/.venv/bin/python" pytest==9.1.1 hypothesis==6.156.6 > "$D/install-tests.log" 2>&1
"$D/.venv/bin/python" -c 'import numpy, pytest, hypothesis; print(numpy.__version__, pytest.__version__, hypothesis.__version__, numpy.__file__)' > "$D/versions.log"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
nice -n 19 "$D/.venv/bin/python" -m pytest -q --disable-warnings -x "$D/.venv/lib/python3.12/site-packages/numpy" > "$D/full-tests.log" 2>&1
