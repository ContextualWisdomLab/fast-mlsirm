# Stable source-path repeat

Prior C/D completed but differ. ELF section analysis shows pip randomized extraction directories in rodata and changed GNU build IDs. This recipe reuses the original commands, extracts the immutable NumPy sdist at /w/b_stable/numpy-2.5.2, and builds the local directory. E/F use separate exclusive host roots but identical container paths for source, backend, OpenBLAS and outputs. Inputs are mounted read-only. Existing C/D are preserved.

Owned SSH execution handle 87273 runs E followed by F only after successful E. E container bf6cc61ed90a was independently inspected running with network none, 2 CPUs and 6 GiB. Do not restart based on an observation timeout. Re-poll that handle or inspect the container/actual process. Final hashes and numerical/notice checks remain pending.

## E native check completed

The notice-repaired E wheel (SHA256 91c7a71387703c23a9bcf953adcc7b0d3ec8f41e6e9a090afaa3a6532c6c8604) was copied to a new exclusive s1 project directory, /data/orca/workspaces/fmls-numpy-stable-E-validation-20260928. uv 0.8.20 created a local CPython3.12.11 venv and installed only that wheel offline with no dependencies. Existing check_installed.py was run unchanged with --sha256, followed by existing check_native.py; both exited zero. Installed NOTICE hash, solve/inverse residuals and loader checks pass. All 20 ELF files have no gfortran/quadmath DT_NEEDED entry. System libgcc-s1/libstdc++6 providers are attributed; provider license acceptance is not claimed. Receipts are retained here. This is one Linux CP312 smoke/static scope, not full NumPy numerical tests, current fast-mlsirm ABI approval, independent compiled-byte repeat acceptance or the published 12-wheel matrix.
