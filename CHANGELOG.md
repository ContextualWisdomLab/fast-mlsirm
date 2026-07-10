# Changelog

## Unreleased

### Changed

- Made the Rust core (`fast_mlsirm._core`) the **primary** numeric path: the
  default `FitConfig.backend` and CLI `--backend` are now `"auto"`, resolving to
  Rust when the compiled extension is available and falling back to the NumPy
  reference otherwise. The verified LSIRM/MLS2PLM neg-loglik, gradient, and
  distance-kernel formulas are ported bit-for-bit; observable outputs are
  unchanged.

### Added

- Added `tests/test_rust_parity.py`, a Rust<->NumPy numerical parity gate that
  asserts agreement to `1e-6` across all five model variants, multiple problem
  sizes, and masked/dense fixtures (observed difference ~1e-13).
- Added a Rust toolchain plus a resolved-default-backend assertion to the
  `python` CI job so the primary Rust path is built and exercised by the suite.
- Added `scripts/release_acceptance.py` to execute a sales-readiness end-to-end
  smoke: simulate, fit (auto + optional rust), diagnostics, and report rendering.
- Added `docs/release_acceptance.md` to document acceptance inputs, outputs, and
  pass criteria.
- Added `docs/enterprise_sales_readiness.md` and `scripts/sales_readiness.py`
  to produce a machine-readable enterprise procurement readiness manifest.

### CI

- Replaced package-only Rust smoke with release-acceptance execution in CI.
- Added an enterprise sales-readiness gate to validate acceptance evidence,
  policy documents, package artifacts, installed-version consistency, and Rust
  backend import proof.

### Documentation

- Updated commercial-readiness and README documents to point to the acceptance
  checklist and execution command.
- Added KRW 2,000,000,000 enterprise sales-review criteria and explicit go/no-go
  evidence requirements.

## 0.1.0 - 2026-07-02

### Added

- MLS2PLM simulation, fitting, diagnostics, and HTML report generation.
- Optional Rust/PyO3 backend exposed as `fast_mlsirm._core`.
- Backend selection through `FitConfig.backend` and `fast-mlsirm fit --backend`.
- Fit summary persistence of the resolved backend.
- Commercial beta readiness documentation, support policy, security policy, and
  release verification checklist.

### Known Limits

- Current estimators are regularized point-estimate JML/MAP-style workflows,
  not Bayesian posterior samplers.
- Ordinal response estimators, sparse/block execution, benchmark automation,
  and posterior predictive checks remain future work.
