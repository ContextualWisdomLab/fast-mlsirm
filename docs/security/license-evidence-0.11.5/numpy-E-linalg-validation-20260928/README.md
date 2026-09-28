# NumPy E linear algebra tests

The exact notice-repaired, Fortran-free Linux CPython 3.12 wheel SHA256 `91c7a71387703c23a9bcf953adcc7b0d3ec8f41e6e9a090afaa3a6532c6c8604` was copied from the prior E validation workspace to a separate s1 project directory, `/data/orca/workspaces/fmls-numpy-E-linalg-validation-20260928`. `run.sh` verifies that hash, creates a local CPython 3.12.11 venv with uv, and installs the wheel and pytest 9.1.1 offline. It then runs the wheel's bundled `numpy/linalg/tests` with one OpenBLAS thread, nice priority 19 and `pytest -q --disable-warnings -x`.

The exact run exited zero: **486 passed, 2 skipped, 1 xfailed, 9 warnings in 42.49 seconds**. The wheel source, install, version and test logs are retained and SHA256-bound in `SHA256SUMS`. Existing E and F build outputs and the prior smoke environment were not modified.

This is an additional numerical check for one local Linux CPython 3.12 wheel. It is not a full NumPy suite, macOS/Windows result, complete ABI or provider-rights approval, original published NumPy wheel acceptance, or the required published 12-wheel release matrix. NumPy remains HOLD.
