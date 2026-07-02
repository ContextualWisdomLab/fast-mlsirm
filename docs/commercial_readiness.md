# Commercial Readiness

## Readiness Position

`fast-mlsirm` is ready for commercial beta deployment by technical users who
understand MLSIRM/IRT workflows and can evaluate model suitability for
their own domain. It is not positioned as a finished regulated decision product
or as a fully managed assessment platform.

## Supported Product Surface

- Python API for simulation, fitting, diagnostics, recovery checks, and report
  rendering.
- CLI workflows for simulation, fitting, fit diagnostics, dimensionality
  diagnostics, response-process diagnostics, and report rendering.
- NumPy reference backend as the default runtime path.
- Optional Rust/PyO3 backend for the fitting objective through
  `fast_mlsirm._core`.
- Backend selection through `FitConfig(backend=...)` and
  `fast-mlsirm fit --backend`.
- Dense response matrices with missing values represented by `NaN`, `-1`, or an
  explicit mask.

## Not Yet Supported

- Sparse/block execution for very large matrices.
- Automated benchmark reporting.
- Posterior predictive checking and Bayesian posterior inference.
- Native ordinal response-model estimation such as GRM, GPCM, or GGUM.
- Hosted dashboards, user management, billing, or enterprise administration.
- Domain-specific clinical, employment, or educational placement decisions.

## Seller Acceptance Checklist (Release Gate)

Before treating a build as sale-ready, verify all items below against the exact
release commit:

- `python3 -m pytest` passes.
- `cargo test --workspace` passes.
- `cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml` passes.
- `python3 -m pip install -e .` or wheel install builds the PyO3 extension.
- `python3 -c "import fast_mlsirm, fast_mlsirm._core"` succeeds.
- `python scripts/release_acceptance.py --out acceptance_check --require-rust` passes and
  writes `acceptance_summary.json`.
- `fit_auto`/`fit_rust` fit summaries are complete and match resolved backend
  paths.
- README, PRD/TRD, `SECURITY.md`, `SUPPORT.md`, `CHANGELOG.md`, and
  `docs/release_acceptance.md` are present and match shipped behavior.
- GitHub CI includes Python tests, Rust core tests, PyO3 crate tests, package
  build validation, wheel metadata checks, and release-acceptance execution.

## Release Gate

Release candidates must not change the model formula, diagnostics semantics, or
estimation scope without a separate model-design review. Packaging, docs, tests,
and examples may change as long as they preserve the formula contract:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * r_pi
r_pi = sqrt(sum_k (xi_pk - zeta_ik)^2 + eps)
```

## Operational Notes

- Source and editable installs require a Rust toolchain because maturin builds
  `fast_mlsirm._core`.
- Installed wheels can use the NumPy backend by default.
- The Rust backend is a dense-matrix backend. It is not a sparse storage layer.
- Real assessment data should be handled under the buyer's own privacy,
  governance, retention, and audit policies.
