# A3 license gate verdict — Linux x86_64 cp312, fast-mlsirm 0.11.5

Status: **provisional for the examined Linux cp312 candidate** (coordinator verdict, user-delegated, 2026-09-26; not legal review). The actual-wheel inventory correction below needs independent re-review. Hosted checks, approval, and release remain pending.
Decision records: coordinator messages msg_60a758c73aad, msg_58e4fad25e39, msg_ee85fb3cb33f, msg_13c9108bac7d, msg_b5aa37b1b0c7, msg_411f3daec1fa.

## Scope
- Source: #2157 head `58b7b23f7a154c8391c130ebc3aab2dde50bb84b`. Its Cargo lockfiles and crate manifests are unchanged from the inventoried `26cd4c83` input; the later changes add attribution, documentation, and the complete libm source notices.
- Candidate artifact checked: `fast_mlsirm-0.11.5-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`
  SHA256 `d8ec1d497763abd943dfc5ba0defa93a67f141b8bab9adf04a02c8d09a9d43bb` (built from the exact head on s1; source archive SHA256 `f8a15e7513f63b765773d5de5fa7c15bc205b3f8e6f24a1cb30097cc923094bb`). `auditwheel` reports `manylinux_2_17_x86_64`; the installed pair passed 10 regression tests.
- Cargo binding graph for `x86_64-unknown-linux-gnu`: 103 rows, **HOLD 0 after the actual-wheel correction below**; union of all targets: 158 rows, HOLD 30 (non-Linux, deferred to 12-wheel expansion). The #2170 parent head `bce6de42` alone has 0.11.4 manifests and does not prove the 0.11.5 result.

## Actual-wheel correction and binding

The previous generator applied all six wheel `License-File` entries to each own crate as candidate license texts. With the real wheel and a measured 64-character SHA256, that gave Linux **101 PERMISSIVE, 2 HOLD**: both own crates rejected five supplemental files as noncanonical, including two unrecognized notices. The corrected common path retains all six files as evidence, tests only the project's `LICENSE` as the own-crate MIT grant, and records the other five as distribution or named third-party attribution. It does not mark those five as canonically verified license texts. An unknown notice owner remains HOLD.

The corrected run uses the exact #2157 source archive and wheel above, the 0.11.5 Cargo input set, `--own-crate-wheel-source-root` pointing to that source, and the Linux target graph. Every declared wheel file must match its source bytes. `LICENSE-THIRD-PARTY` must match the tracked snapshot SHA256 `e6d920bd6bdd392cc902a5a54fe365f48fdadb1b3762e857fddee1e4dea6ca26`, its source inventory SHA256 `621e4ca2e51b460c832438c1ec04cb679ca44c2dddad396d08103c2e99590d28`, and all 101 third-party rows in the target graph. The two own crate rows are version 0.11.5, PERMISSIVE, with no HOLD reasons; all 103 Linux rows are PERMISSIVE and completeness gaps are empty. Corrected inventory SHA256: `d8516b5a9f515d545c877e792091fde29e565f79a994f207adcfca13879dba56` at s1 `runs/fmls-a3-wheel-rca-58b7b23f-20260926/fixed-bound/inventory.json`. This is local evidence pending independent review, not a hosted gate result.

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
