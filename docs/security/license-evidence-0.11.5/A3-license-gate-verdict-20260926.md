# A3 license gate verdict — Linux x86_64 cp312, fast-mlsirm 0.11.5

Status: **complete for the examined Linux cp312 candidate** (coordinator verdict, user-delegated, 2026-09-26; not legal review). Hosted checks, independent review, and release remain pending.
Decision records: coordinator messages msg_60a758c73aad, msg_58e4fad25e39, msg_ee85fb3cb33f, msg_13c9108bac7d, msg_b5aa37b1b0c7, msg_411f3daec1fa.

## Scope
- Source: #2157 head `58b7b23f7a154c8391c130ebc3aab2dde50bb84b`. Its Cargo lockfiles and crate manifests are unchanged from the inventoried `26cd4c83` input; the later changes add attribution, documentation, and the complete libm source notices.
- Candidate artifact checked: `fast_mlsirm-0.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  SHA256 `d8ec1d497763abd943dfc5ba0defa93a67f141b8bab9adf04a02c8d09a9d43bb` (built from the exact head on s1; source archive SHA256 `f8a15e7513f63b765773d5de5fa7c15bc205b3f8e6f24a1cb30097cc923094bb`). `auditwheel` reports `manylinux_2_17_x86_64`; the installed pair passed 10 regression tests.
- Cargo binding graph for `x86_64-unknown-linux-gnu`: 103 rows, **HOLD 0**; union of all targets: 158 rows, HOLD 30 (non-Linux, deferred to 12-wheel expansion).

## Acceptance bases (Linux graph, 103 rows)
canonical verifier text 21; byte-identical condition clauses (reviewed hash) 68; SPDX matching template 4;
upstream-vcs text at the published commit 2; own crates via published-wheel LICENSE 2; reviewed pointer notices 4;
hash-pinned documented exceptions 2 (libm 0.2.16 notice-preserving permissive; cfg_aliases 0.2.2 third-party MIT attribution).
An independent strict SPDX-template re-judgment found verifier diff 0; its 4 disagreements (cfg_aliases, libm,
matrixmultiply, ndarray) are all accepted by explicit verdicts above, not by the template criterion.

## Attribution in the candidate wheel
`dist-info/licenses/`: LICENSE, LICENSE-THIRD-PARTY (101 crates, sha256 `d46f307a…`), NOTICE, libm LICENSE and complete source notices (sha256 `9e949a13f66c0f9b60b73b54e8ab2940ccff92d704c46c53103b1028e2cc75ba`), cfg_aliases NOTICES.md. The libm notice bytes match the exact-head source.

## Outside this gate
- NumPy is an unbundled external runtime dependency (official wheel `3cdec01f…` vendors libgfortran/libquadmath; not a fast-mlsirm bundling violation). Research consumption uses a Fortran-free NumPy build.
- Python non-shipping tooling (atheris, colorama, hypothesis x2, numpy text files, packaging, pygments, sortedcontainers): tracked as a separate license-hygiene item; none is GPL/LGPL/AGPL (strongest: MPL-2.0).
