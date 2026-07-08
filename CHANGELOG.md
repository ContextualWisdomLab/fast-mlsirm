# Changelog

## Unreleased

### Added

- GPGPU acceleration of the negative-log-likelihood and gradient hot path inside
  the Rust core via [wgpu](https://github.com/gfx-rs/wgpu) (MIT/Apache-2.0),
  exposed as a device sub-option of the Rust backend rather than a separate
  compute-backend axis. Select with `FitConfig(backend="rust", rust_device=...)`
  or `fast-mlsirm fit --backend rust --rust-device {auto,cpu,gpu}`; the GPU path
  falls back to the identical CPU implementation at runtime when no GPU adapter
  is available. Added `FitResult.rust_device` and `fit_summary.json`
  `rust_device` provenance, plus numerical-parity tests asserting the Rust
  device paths match the NumPy reference.
- Added `docs/papers/` with the CC BY 4.0 paper (Wu et al., 2021, arXiv:2108.11579)
  grounding fast, accelerator-friendly IRT estimation.
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
