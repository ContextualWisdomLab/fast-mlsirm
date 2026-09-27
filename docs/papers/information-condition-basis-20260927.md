# Symmetric information condition diagnostics

Source audit: ContextualWisdomLab/late-life-anxiety-reanalysis commit 63984c25075a1deef32fc32fd00c0748cd3d2c1a, evaluations/runs/20260927-information-condition-sources/report.md.

The sources actually opened are the official LAPACK Users’ Guide, not a journal paper:
- https://www.netlib.org/lapack/lug/node75.html (How to Measure Errors, table 4.2 and condition/RCOND paragraphs).
- https://www.netlib.org/lapack/lug/node89.html (symmetric orthogonal eigendecomposition and ANORM).
- https://www.netlib.org/lapack/lug/node90.html (small-eigenvalue relative accuracy limitation).

The symmetric 2-norm condition formula max(abs(eigenvalues))/min(abs(eigenvalues)) is derived from the norm definition and orthogonal eigendecomposition, not represented as a quotation from a paper. It describes the symmetrized input used by the existing second_order_test. It is independent of positive-definiteness. Singular computed spectra return infinity and zero reciprocal; floating-point overflow/underflow can produce the same limits. No conditioning acceptance threshold is introduced.

A runnable regression exposed the existing Jacobi absolute-tolerance problem: a 45-degree rotation of diag(1,3), scaled by 1e-300, returned condition 1 instead of 3. Normalize the matrix by its largest absolute entry before the shared Jacobi iteration and rescale eigenvalues afterwards. This is an implementation choice supported by the failing synthetic test, not a claim that the cited guide or a journal paper prescribes this patch. Existing pseudoinverse callers share this path. Nonfinite restored eigenvalues are rejected. Small eigenvalues may still have poor relative accuracy; no LAPACK error guarantee is claimed for this implementation.

Validation:
- Before normalization: standalone rotation/scale test failed with condition 1 at scale 1e-300.
- rustc --edition 2021 --test crates/mlsirm-core/src/inference.rs followed by binary execution: 9 passed after repair, including existing inference tests.
- Python wrapper with mocked native results: 1 passed, 1 deselected.
- Installed API against the older b4661fe7 candidate: fails with KeyError condition_number_2. No skip fallback. New native build and installed tests remain pending.

Full crate/PyO3 build, hosted gates, non-author review and immutable release/install provenance remain required. No actual study matrix or reported study value is produced here. Related: #2200.

## Shared-caller regression

The Jacobi repair also affects vcov_from_hessian through pseudoinverse_symmetric. A real Rust call on rank-one matrices with common scales 1, 1e-300 and 1e300 preserves the known pseudoinverse entries (checked after multiplying by the input scale to avoid test overflow). Standalone inference module execution now passes 10 tests. This synthetic regression does not establish actual study ACOV validity or installed PyO3 acceptance. The current-head hosted checks were queued and both PRs had no submitted non-author review at the 2026-09-27 observation; queued checks are not passing evidence.
