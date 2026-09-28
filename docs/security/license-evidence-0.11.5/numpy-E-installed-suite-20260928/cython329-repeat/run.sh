#!/usr/bin/env bash
set -euo pipefail
D=/data/orca/workspaces/fmls-numpy-E-cython329-suite-20260928
W=/data/orca/workspaces/fmls-numpy-stable-E-validation-20260928/input/numpy-2.5.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
B=/data/orca/workspaces/fmls-numpy-pinned-backend-build-20260928/input/build-deps
C="$D/cython-3.2.9-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl"
UV=/home/seongho/.local/bin/uv
PY=/home/seongho/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
trap 'printf "%s\n" "$?" > "$D/exit.status"' EXIT
"$UV" venv --python "$PY" "$D/.venv" > "$D/venv.log" 2>&1
"$UV" pip install --offline --python "$D/.venv/bin/python" --no-deps "$W" > "$D/install-numpy.log" 2>&1
"$UV" pip install --offline --python "$D/.venv/bin/python" pytest==9.1.1 hypothesis==6.168.0 > "$D/install-tests.log" 2>&1
"$UV" pip install --offline --python "$D/.venv/bin/python" --no-deps "$B/meson-1.12.1-py3-none-any.whl" "$B/ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl" "$C" > "$D/install-backend.log" 2>&1
"$D/.venv/bin/python" -c 'import numpy, pytest, hypothesis, Cython; print(numpy.__version__, pytest.__version__, hypothesis.__version__, Cython.__version__, numpy.__file__)' > "$D/versions.log"
export PATH="$D/.venv/bin:$PATH" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
nice -n 19 "$D/.venv/bin/python" -m pytest -q --disable-warnings -x "$D/.venv/lib/python3.12/site-packages/numpy" > "$D/full-tests.log" 2>&1
