# A3 license gate verdict — Linux x86_64 cp312, fast-mlsirm 0.11.5

Status: **complete for shipped artifacts** (coordinator verdict, user-delegated, 2026-09-26; not legal review).
Decision records: coordinator messages msg_60a758c73aad, msg_58e4fad25e39, msg_ee85fb3cb33f, msg_13c9108bac7d, msg_b5aa37b1b0c7, msg_411f3daec1fa.

## Scope
- Source: 0.11.5 prep `26cd4c83` (= #2157 `e1fa5189` + version 0.11.5 + uv.lock numpy 2.5.2).
- Shipped artifact checked: `fast_mlsirm-0.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  SHA256 `2c48a45752a1d9301923c54c264cd166e56505ce543f02d193635edcfb1b0e6f` (built from `a7b99279`).
- Cargo binding graph for `x86_64-unknown-linux-gnu`: 103 rows, **HOLD 0**; union of all targets: 158 rows, HOLD 30 (non-Linux, deferred to 12-wheel expansion).

## Acceptance bases (Linux graph, 103 rows)
canonical verifier text 21; byte-identical condition clauses (reviewed hash) 68; SPDX matching template 4;
upstream-vcs text at the published commit 2; own crates via published-wheel LICENSE 2; reviewed pointer notices 4;
hash-pinned documented exceptions 2 (libm 0.2.16 notice-preserving permissive; cfg_aliases 0.2.2 third-party MIT attribution).
An independent strict SPDX-template re-judgment found verifier diff 0; its 4 disagreements (cfg_aliases, libm,
matrixmultiply, ndarray) are all accepted by explicit verdicts above, not by the template criterion.

## Attribution shipped
`dist-info/licenses/`: LICENSE, LICENSE-THIRD-PARTY (101 crates, sha256 `d46f307a…`), NOTICE, libm LICENSE and source notices, cfg_aliases NOTICES.md.

## Outside this gate
- NumPy is an unbundled external runtime dependency (official wheel `3cdec01f…` vendors libgfortran/libquadmath; not a fast-mlsirm bundling violation). Research consumption uses a Fortran-free NumPy build.
- Python non-shipping tooling (atheris, colorama, hypothesis x2, numpy text files, packaging, pygments, sortedcontainers): tracked as a separate license-hygiene item; none is GPL/LGPL/AGPL (strongest: MPL-2.0).
