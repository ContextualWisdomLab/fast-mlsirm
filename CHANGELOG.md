# Changelog

## Unreleased

### Added

- **Marginal (MMLE-EM) estimation for the full latent-space family.**
  `fit(estimator="mmle")` now fits `MIRT`/`MLS2PLM`/`MLSRM` (and `ULS2PLM`/
  `ULSRM` under a population structure) by Bock-Aitkin-style marginal EM:
  person latents `(theta, xi)` are integrated over Gauss-Hermite grids —
  tractable via the simple-structure conditional factorization — with a
  Fisher-preconditioned GEM M-step and the Jeon et al. (2021) LSIRM priors as
  MAP penalties (`PenaltyConfig::lsirm_prior`). Rust core
  (`mlsirm_core::marginal`) with a NumPy mirror
  (`fast_mlsirm.estimators.marginal`) held to 1e-9 end-of-run parity
  (`tests/test_marginal_parity.py`); design and paper basis in
  `docs/mmle_marginal_lsirm_design.md`.
- **Estimation-level multigroup and multilevel population structures** for the
  marginal estimator: `fit(..., group_id=...)` (Bock-Zimowski group trait
  means/SDs, common items, pinned reference group) and
  `fit(..., cluster_id=...)` (Fox-Glas random intercept, `sigma_u`/ICC
  estimated). Results surface on `FitResult.population` and persist through
  `save_fit_result`; the CLI `fit` command gains `--estimator`, `--group-id`,
  `--cluster-id`, `--q-theta`, `--q-xi`, `--q-u`, and `--tolerance`.
- **wgpu E-step kernels for the marginal estimator**
  (`mlsirm_core::gpu_marginal`): the E-step hot path runs in f32 on the GPU
  with the same race-free slot-ownership reduction as the JML kernels, cutting
  a 31k-person multilevel E-step iteration from ~110 s (CPU f64) to ~5 s on a
  laptop RTX 3050 Ti; the M-step and final EAP pass stay on the CPU in f64,
  and hosts without an adapter fall back to the CPU path unchanged.
- **Likelihood-based fit statistics** (`fast_mlsirm.fitstats`): Orlando-Thissen
  S-X² via the Lord-Wingersky recursion generalized to the joint `(theta, xi)`
  grid (chi-square tail without SciPy), Benjamini-Hochberg FDR control,
  Drasgow `l_z` and Snijders `l_z*` person fit with the MAP `r_0` correction,
  and infit/outfit at the marginal EAPs.
- **M2 limited-information goodness-of-fit** (`fast_mlsirm.fitstats.m2`;
  Maydeu-Olivares & Joe 2005/2006, Cai & Hansen 2013): the M2 statistic on the
  univariate + bivariate residual margins, its df and χ² tail p-value, the
  RMSEA2 approximate-fit index with a 90% noncentral-χ² confidence interval,
  and the bivariate SRMSR (Maydeu-Olivares 2013). Every model-implied margin
  (and the up-to-4th-order entries of the multinomial residual covariance
  `Xi_2`) is computed exactly by the local-independence factorization over the
  `(theta, xi)` node set — `pi_S = Σ_c w_c ∏_{i∈S} P_i(c)` — the same
  factorization the E-step already uses (Cai-Hansen); the derivative matrix
  `Delta_2` is central-differenced from the node moments and the quadratic form
  is evaluated through one Cholesky of `Xi_2` (never an explicit inverse). Rust
  core (`mlsirm_core::fitstats::m2_rmsea2`, kind-aware) with a NumPy reference
  held to 1e-6 parity; well-specified-vs-local-dependence calibration tests in
  both suites.
- **Item screening pipeline** (`fast_mlsirm.select_items`): iterative
  fit → flag → remove → refit with sparse / S-X²-BH / mean-square band /
  low-discrimination / map-isolation flags, an `l_z*` person screen, a
  per-dimension item floor, and a full audit trail.
- **Serving bundle + frozen-parameter scoring** (`fast_mlsirm.serving`):
  schema-versioned JSON bundle of the calibrated item parameters and
  population block, and `score_respondents()` EAP scoring of new response
  payloads with items frozen — the fixed-parameter serving pattern used by
  the downstream importance-assessment API. `fast-mlsirm score` scores a JSON
  payload (or `.npy` matrix) against a bundle from the command line.

- **QMC-EM and MC-EM integration rules** for the marginal estimator
  (`FitConfig(xi_rule="qmc"|"mc", xi_points=..., xi_seed=...)`): the
  latent-space integral runs on Halton low-discrepancy points (randomized-QMC
  shift optional; Jank 2005) or seeded Monte Carlo draws (Wei & Tanner 1990;
  Meng & Schilling 1996) instead of the tensor Gauss-Hermite grid — enabling
  `latent_dim > 3` and better error scaling per node. Both constructions are
  deterministic and bit-mirrored across the Rust/NumPy backends.
- **Rust scoring module** (`mlsirm_core::scoring`, exposed via
  `_core.score_bank_eap` / `score_bank_map` / `eapsum_tables`): EAP
  (Bock & Mislevy 1982), MAP (posterior Newton with observed-information
  SEs), and summed-score EAP conversion tables via the Lord-Wingersky
  recursion (Thissen et al. 1995; Cai 2015), all under per-dimension
  `N(mean_d, sd_d^2)` priors that cover single, multigroup
  (`mu_g, sigma_g`) and multilevel populations (conditional
  `N(u_hat_c, 1)` or marginal `N(0, sqrt(1 + sigma_u^2))`).
  `score_respondents(..., method="eap"|"map"|"eapsum", prior=...)` and the
  bundle's embedded `eapsum_tables` expose these to serving.
- **Fit statistics moved to the Rust core** (`mlsirm_core::fitstats`): S-X²,
  Benjamini-Hochberg, `l_z`/`l_z*`, infit/outfit now compute in Rust
  (`fast_mlsirm.fitstats` delegates; the NumPy bodies remain the parity
  reference/fallback). S-X² gains the `rms_residual` practical-significance
  effect size (Sinharay & Haberman 2014) and `select_items` gates its flag on
  `sx2_min_effect`; the mean-square gate now uses infit only (outfit is
  reported, not gating — it explodes under very low pass rates); the person
  screen threshold is configurable and the Snijders `r_0` correction is
  centered on the population prior mean (cluster intercepts / group means).
- **Fixed Item Parameter Calibration** (`fit(..., anchors=...)`): anchored
  items stay frozen (optionally `tau` too) while new items and a freed
  population mean/SD are estimated — the multiple-cycle prior-update (MWU-MEM
  style) variant Kim (2006) found robust; latent-space orientation inherits
  from the anchors (no PCA re-alignment). **Concurrent calibration** is the
  existing multigroup path with structural missingness (Hanson & Béguin
  2002), covered by a dedicated recovery test.

### Changed

- `estimator="mmle"` with a spatial/multidimensional model now fits (routed to
  the marginal estimator) instead of raising `NotImplementedError`; plain
  `ULS2PLM`/`ULSRM` without a population structure keep the legacy
  unidimensional fast path and its exact previous behavior.

- Exposed the Rust MMLE-EM estimator (`mlsirm_core::mmle::fit_mmle_2pl`) through
  the PyO3 binding as `fast_mlsirm._core.fit_mmle_2pl`, so
  `fit(estimator="mmle")` now runs on the Rust core when the extension is built
  (previously it always fell back to the NumPy reference). To keep the two
  backends statistically equivalent, the Rust core's Gauss-Hermite table was
  aligned from 21 to 41 nodes, bit-identical to the NumPy reference's default
  `hermegauss(41)` quadrature; `tests/test_rust_parity.py` gains MMLE parity
  tests (a/b/theta agreement at the shared EM optimum, measured ~1e-8).

- Made the Rust core (`fast_mlsirm._core`) the **primary** numeric path: the
  default `FitConfig.backend` and CLI `--backend` are now `"auto"`, resolving to
  Rust when the compiled extension is available and falling back to the NumPy
  reference otherwise. The verified LSIRM/MLS2PLM neg-loglik, gradient, and
  distance-kernel formulas are ported bit-for-bit; observable outputs are
  unchanged.

### Added

- GPGPU acceleration of the negative-log-likelihood and gradient hot path inside
  the Rust core via [wgpu](https://github.com/gfx-rs/wgpu) (MIT/Apache-2.0),
  exposed as a device sub-option of the Rust backend rather than a separate
  compute-backend axis. Select with `FitConfig(backend="rust", rust_device=...)`
  or `fast-mlsirm fit --backend rust --rust-device {auto,cpu,gpu}`; the GPU path
  falls back to the identical CPU implementation at runtime when no GPU adapter
  is available. Added requested-device provenance on `FitResult.rust_device`
  and in `fit_summary.json`, plus numerical-parity tests asserting the Rust
  device paths match the NumPy reference.
- Added `docs/papers/README.md` with a citation and canonical link for Wu et al.
  (2021, arXiv:2108.11579), grounding fast, accelerator-friendly IRT estimation
  without vendoring the PDF into the repository.
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
- Added aFIPC-style fixed-item calibration diagnostics and
  `diagnose-fixed-item-calibration` to select candidate probability tensors
  with kaefa-style item-fit penalty evidence.

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
- Updated the Figma product design packet with Information Architecture,
  화면정의서, key screen, wireframe, and user stories for fixed-item
  calibration review.

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
