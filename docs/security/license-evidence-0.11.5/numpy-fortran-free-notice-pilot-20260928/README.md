# Local Fortran-free NumPy NOTICE repair

The source candidate is the **rep_A** wheel on s1, SHA256
`c6fbea224227ebfcb80d694be15f74c72a06c901833f3f073df3405ade4e07a9`.
The identically named file at the parent folder is the official wheel,
`3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8`;
it is not the candidate. Both originals remain unchanged.

A copied candidate was repacked twice with the staged source-derived
OpenBLAS/GotoBLAS/LAPACK/LAPACKE notices. METADATA declares the added
License-File and RECORD covers every file. Original native and source members
are byte-identical. Both output ZIPs are byte-identical, SHA256
`011ba92963c8fd230f3e81ac8f20af6c91e08bb139f0497cdb6ede1cc95c9a06`.
The exact original source/archive and four source notice hashes are recorded
in source-hashes.txt. This is repeat **repacking**, not repeat compilation.

To reproduce, copy the hash-bound rep_A wheel into candidate-input/ beside
repair_notice.py and run `python3 repair_notice.py` in a fresh copy of this
folder. It creates A/ and B/ exclusively, validates the original and repaired
RECORD, preserves all native/source bytes, and compares the two outputs.
Current binaries are retained under /tmp/fmls-numpy-notice-pilot-20260928/;
large wheels are not committed. No original artifact was modified.

This repairs the missing embedded notice only. No license-policy PASS,
independent build provenance, full NumPy tests, consumer provider acceptance,
current-head fast-mlsirm numerical approval, hosted run or published matrix is
claimed. Existing GNU-family and SDK/source-rights HOLDs remain.

The repaired wheel was installed offline with uv in a new project-local
CPython3.12.11 environment on s1. The declared installed NOTICE hash matches;
linear solve and inverse residual smoke checks pass; current process maps
contain no libgfortran/libquadmath. Provider attribution checks `/usr/lib`
and a `/lib` alias only when they refer to the same physical file. libgcc_s
is owned by libgcc-s1; libstdc++ by libstdc++6. This is observed s1 package
ownership only, not license-policy approval or a consumer/platform-wide claim.
The runnable check and exact JSON receipt are retained alongside this note.
