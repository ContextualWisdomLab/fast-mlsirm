# Commercial Readiness

## Readiness Position

`fast-mlsirm` is ready for commercial beta deployment by technical users who
understand MLSIRM/IRT workflows and can evaluate model suitability for
their own domain. It is not positioned as a finished regulated decision product
or as a fully managed assessment platform.

For KRW 2,000,000,000 enterprise sales review, use this document together with
`docs/enterprise_sales_readiness.md` and `docs/20b_product_readiness.md`. That
higher gate requires procurement evidence, release artifacts, support/security
scope, Product Design and Figma packet evidence, Data Analytics ROI/benchmark
evidence, and a machine-readable `sales_readiness_manifest.json`.

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
- Automated benchmark evidence reporting from release-acceptance timing.

## Not Yet Supported

- Sparse/block execution for very large matrices.
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
- `python scripts/build_benchmark_report.py --acceptance acceptance_check/acceptance_summary.json --out acceptance_check/benchmark`
  writes `benchmark_report.json` and `benchmark_report.html`.
- `fit_auto`/`fit_rust` fit summaries are complete and match resolved backend
  paths.
- README, PRD/TRD, `SECURITY.md`, `SUPPORT.md`, `CHANGELOG.md`, and
  `docs/release_acceptance.md` are present and match shipped behavior.
- GitHub CI includes Python tests, Rust core tests, PyO3 crate tests, package
  build validation, wheel metadata checks, and release-acceptance execution.
- `python scripts/sales_readiness.py --acceptance acceptance_check/acceptance_summary.json --dist dist --require-rust --check-import`
  passes and writes `sales_readiness_manifest.json` when verifying a built
  artifact.
- `python scripts/sales_readiness.py --acceptance acceptance_check/acceptance_summary.json --dist dist --require-rust --require-20b-product --benchmark-report acceptance_check/benchmark/benchmark_report.json --require-benchmark-report --check-import`
  passes when positioning the release for KRW 2,000,000,000 procurement review.
- `python scripts/build_buyer_packet.py --acceptance acceptance_check/acceptance_summary.json --sales-readiness acceptance_check/sales_readiness_manifest.json --dist dist --benchmark-report acceptance_check/benchmark/benchmark_report.json --out buyer-evidence-packet`
  creates `buyer_evidence_manifest.json` and
  `fast_mlsirm_buyer_evidence_packet.zip`, plus
  `buyer_evidence_report.html`, when a portable procurement packet is part of
  the offer.

## Enterprise Sales Gate

The KRW 2,000,000,000 sales-readiness standard is not just a smoke test. A candidate
must be able to show:

- release acceptance evidence generated from the exact artifact;
- built wheel and source distribution files;
- installed package import proof, including the Rust backend if sold as part of
  the package;
- explicit support, security, non-goal, and formula-contract boundaries;
- buyer demo storyboard, Figma packet with Code Connect disabled, ROI evidence,
  benchmark manifest, generated benchmark report, and synthetic demo package;
- a generated `sales_readiness_manifest.json` with no failed checks.
- a generated buyer evidence packet with SHA256 digests and standalone HTML
  review when procurement asks for a single reviewable artifact bundle.

## Security and Support Boundaries

Security scope is documented in `SECURITY.md`. Support scope is documented in
`SUPPORT.md`. Both files are required evidence for the enterprise sales gate and
must match the exact package behavior being offered.

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
