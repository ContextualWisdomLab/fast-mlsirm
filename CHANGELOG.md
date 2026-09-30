Warning: truncated output (original token count: 137725)
Total output lines: 6637

# Changelog

## Unreleased

### Added

- Saved multiple-group bifactor and two-tier GRM fits can now produce
  per-person conditional `l_z`, posterior trait estimates, observed-item
  counts, and caller-threshold flags. Optional seeded model resampling gives
  an empirical lower-tail probability; no cross-loading `l_z*` or normal-null
  calibration is claimed (#2116).

### Changed

- Standalone report JSON/CSV export regions now apply atomic text selection
  only for fine-pointer devices. Touch and coarse-pointer users retain native
  partial selection; keyboard focus, exact-value tables, JSON/CSV payloads,
  escaping, and reduced-motion behavior are unchanged. This is a pointer
  convenience, not a dedicated keyboard or assistive-technology copy action
  (#2304).

- Polytomous person fit now requires convergence by default, including for
  duck-typed fits with unknown convergence. `allow_unconverged=True` permits
  diagnostic use only for legacy or duck-typed fits and marks the result
  `valid_person_fit=False`, `diagnostic_only=True`; unconverged `PolyFipcFit`
  always raises. Provenance retains convergence and termination fields (or
  `"unknown"`). `PolytomousFit` retains its established standard-normal
  EAP grid even with non-default `prior_mean`/`prior_sd`, which affect only
  the `r0` correction. `PolyFipcFit` uses its fitted focal prior for both.

<!-- BEGIN AUTHORITATIVE CHANGELOG FRAGMENTS -->
### Changed

#### Release cut 0.11.4

- Project version is bumped to 0.11.4 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.11.4] - 2026-09-18` release section, headlined by the PyPI package-description
  boundary repair (#1993): `README.md` no longer carries internal commercial-boundary
  vocabulary or repo-relative links that 404 on the registry page, so the corrected
  immutable description can ship after the 0.11.3 page. The section also folds the
  two-tier / multi-primary Oakes SE and streaming E-step memory work (#1992) and the
  support-policy / bifactor quadrature test repairs that cleared red `main`.
- This cut removes the standing predecessor note `release-0.11.3-cut.md`, whose
  substance is permanently recorded in the `[0.11.3] - 2026-09-18` section and
  in git history.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.
<!-- END AUTHORITATIVE CHANGELOG FRAGMENTS -->


## [0.11.4] - 2026-09-18

### Added

#### Two-tier / multi-primary Oakes SE and streaming E-step memory (#1992)

- Add `two_tier_oakes_se` (Rust + PyO3 + Python) for confirmatory multi-primary
  / G+W method-factor GRM observed-information SEs via Oakes (1999, eq. 6,
  p. 480), with the same fail-loud non-PD contract as `bifactor_oakes_se`.
- Stream the two-tier E-step / EAP / M-step over the full product Gauss–Hermite
  primary grid without materializing per-item `n_grid * q_specific * n_cat`
  log-prob tables or `n_grid * q_specific * n_primary` node tensors (#1992
  memory), keeping caller-controlled node counts (no silent caps; #1929).
- mirt 1.46.1 Oakes cross-check fixture on the stage-4 dataset
  (`seed=20260917`, `quadpts=15`).

### Changed

#### README public-description boundary (#1993)

- `README.md` is the PyPI `long_description`, so it no longer carries internal
  commercial-boundary vocabulary. The `Commercial Readiness` section — with the
  enterprise sales gate, the KRW 2,000,000,000 product gate, buyer packet,
  procurement due-diligence, PR queue governance, Figma evidence sync links, and
  the multi-step evidence build transcript — is replaced by a `Project Status`
  section that states scope, release verification, and the security, support,
  changelog, and ADR entry points. The same evidence machinery is unchanged and
  stays documented in `docs/commercial_readiness.md` and
  `docs/release_acceptance.md`.
- Repo-relative README links now resolve to absolute GitHub URLs. Only `LICENSE`
  and the Python sources ship in the distribution, so `docs/`, `SECURITY.md`,
  `SUPPORT.md`, and `CHANGELOG.md` links were dead on the PyPI project page.

### Fixed

#### README public-description boundary (#1993)

- `scripts/sales_readiness.py` gained a `public_boundary:README.md` check that
  fails the gate when internal commercial, procurement, buyer, or monetary-target
  vocabulary reappears in the published package description. The required
  commercial tokens it used to demand from `README.md` are now required in
  `docs/commercial_readiness.md` and `docs/enterprise_sales_readiness.md`, where
  that language belongs.

#### Red main: support-policy version and a stale quadrature assertion (#1993)

- `SECURITY.md` and `SUPPORT.md` still named `0.10.x` as the supported pre-1.0
  line after the 0.11 releases, so
  `tests/test_support_policy_version_contract.py` failed on `main` and blocked
  every PR's `python` check. Both now name `0.11.x`.
- `tests/test_bifactor_oakes.py::test_rejects_out_of_range_caller_arguments`
  still asserted that `q_general=5` is rejected. #1929 deliberately removed the
  Gauss-Hermite node-count cap — `SUPPORTED_Q` membership became a plain
  `q >= 1` check — and updated the same assertion in
  `tests/test_bifactor_grm.py` and `tests/test_bifactor_multigroup.py` but
  missed this file. The case now uses `q_general=0`, which is still invalid,
  and carries the same `#1929` note as its siblings.


## [0.11.3] - 2026-09-18

### Fixed

#### Bifactor GPU E-step Metal/WebGPU workgroup-dimension limit (#1987)

- Split bifactor reduced E-step compute dispatches across `(x, y, z)` using the
  adapter's runtime `max_compute_workgroups_per_dimension` so Apple Metal no
  longer panics when a 1-D workgroup count exceeds 65535 (AC late-life
  multigroup bootstrap at q=241 required 141573 groups on `reduce_counts_blk`).
- Query `max_storage_buffer_binding_size` / `max_buffer_size` before allocating
  E-step buffers and fall back to the f64 CPU path when they do not fit; no
  hardcoded workgroup or byte caps.
- Replace the WGSL zero-mass log-weight sentinel `-1e300` with an f32-representable
  `-1e37` so `create_shader_module` succeeds on Metal (WGSL rejects the abstract
  literal inside an `f32` comparison).
- Re-enable the `mlsirm-core` default `gpu` feature on the PyO3 cdylib (it had been
  disabled via `default-features = false` in a WIP salvage commit), so `device="gpu"`
  again reaches the wgpu kernels instead of always falling back to CPU.
- Extend study-precision CPU/GPU parity coverage with a q=481 leg and a wide-item
  q=241 Metal 2-D dispatch leg gated by `STAGE5_HIGH_Q=1`.


## [0.11.2] - 2026-09-18

### Added

#### Moderated (simple) slopes on the H1–H5 OLS design (#1985)

- Rust + PyO3 + Python helpers for Aiken–West / Hayes pick-a-point slopes on
  the length-10 `Y ~ X*W*Z + X*E` design: `xwz_e_design_row`,
  `design_row_dot`, `conditional_slope`, and `slope_difference`, reusing the
  existing HC sandwich `linear_contrast` path for SEs (no SciPy / Rscript).
- H1–H5 parity fixtures under `tests/data/regression_h1_h5/`
  (`beta_vcov.npz` aggregates plus coefficient/contrast CSVs) and
  `tests/test_moderated_slopes_h1_h5_parity.py` asserting estimate and HC3 SE
  to atol `1e-6` against `library_regression_h1_h5_contrasts.csv`.


## [0.11.1] - 2026-09-18

### Changed

#### ADR-0028 naming/defaults applied to `dif`, `deltaplot`, and `polytomous` (#1962)

- **`fast_mlsirm.deltaplot.delta_plot`: `alpha` and `max_iter` are now
  required keyword arguments** (previously defaulted to `0.05` and `10`).
  Neither value has a documented source in this repository (ADR-0028 rule 2
  for `alpha`, a decision threshold; rule 1 for `max_iter`, an
  iteration/convergence control), so the package no longer ships a default
  it cannot defend. Every in-repo call site was updated to pass the old
  values explicitly.
- **`fast_mlsirm.dif`: `raju_area`'s `alpha` and `sibtest`'s `fdr_q`/`j_min`
  are now required keyword-only arguments** (previously `0.05`, `0.05`,
  `5`), for the same reason (ADR-0028 rule 2: no cited source for these
  specific cutoffs).
- **`fast_mlsirm.dif`: the renamed DIF entry points also drop the same
  unsourced defaults on their new names** (old aliases keep accepting the
  old default for one minor release): `detect_dif_mantel_haenszel`'s
  `fdr_q`; `detect_dif_mantel_haenszel_purified`'s `fdr_q`, `max_rounds`,
  `min_anchor_items`; `detect_dif_logistic`'s `fdr_q`, `max_iter`;
  `detect_dif_logistic_purified`'s `fdr_q`, `max_iter`, `max_rounds`,
  `min_anchor_items`; `detect_dif_breslow_day`'s `fdr_q`.
- **`fast_mlsirm.polytomous`: unsourced defaults removed from fit/scoring
  entry points.** `fit_lsirm_polytomous` (`max_iter`, `model`, `q_theta`,
  `q_xi`, `tol`), `fit_nominal_polytomous` (`max_iter`, `q_theta`, `tol`),
  `fit_poly_fipc` (`max_iter`, `q_theta`, `tol`), `fit_polytomous`
  (`max_iter`, `model`, `q_theta`, `tol`), `m2_polytomous` (`q_theta`), and
  `score_polytomous` (`q_theta`) now require these arguments explicitly —
  same #1929 quadrature-node-count rule and ADR-0028 rules 1/2/4 already
  applied to the sibling `dif_polytomous*` functions in #1958/#1960. The
  renamed functions `simulate_cat_polytomous` (`q_theta`, `se_threshold`,
  `seed`), `compute_item_fit_polytomous` (`min_expected`, `q_theta`),
  `diagnose_local_dependence_polytomous` (`q_theta`),
  `compute_person_fit_polytomous` (`flag_threshold`, `q_theta`), and
  `compute_u3_cutoff_polytomous` (`alpha`, `n_rep`, `seed`) drop the same
  category of default under their new name; old aliases keep the old
  default for one minor release.
- All in-repo call sites (package, tests, docs, examples) that relied on a
  removed default were updated to pass the old value explicitly.
- **Not yet mirrored on the PyO3 side (Python-only change; follow-up
  needed).** The PyO3 entry points backing `logistic_dif`/
  `logistic_dif_purified` (`fdr_q`, `max_iter`) keep their own Rust-side
  defaults (`fast_mlsirm._core` module functions, per
  `docs/api/renames-and-defaults-20260917.csv`'s `pyo3` rows) — mirroring
  the Python-side default removal into the PyO3 binding is a separate,
  Rust-build-required change tracked for a follow-up PR rather than done
  here (this PR is Python-only per its scope).

#### ADR-0028 naming/defaults applied to the bifactor modules (#1963)

- **`fit_bifactor_grm` (`bifactor_grm.py`) no longer defaults `max_iter`,
  `tol`, `n_starts`, or `seed`.** All four are now required caller
  arguments: `max_iter`/`tol` are unsourced iteration/convergence precision
  controls, `n_starts` is an unsourced replicate count, and `seed` is a
  fixed stochastic seed baked into the previous default — the same ADR-0028
  reasoning already applied to `dif_polytomous_purified` (#1958/#1960) and
  the `q_general`/`q_specific` node counts (#1929) on this same function.
- **`fit_bifactor_grm_fipc` (`bifactor_grm.py`) no longer defaults `max_iter`
  or `tol`.** Same reasoning as above; `q_general`, `q_specific`,
  `newton_iter`, `ridge`, and `estimate_specific_vars` are unchanged
  (out of ADR-0028 policy scope per the decision table).
- **`fit_bifactor_grm_multigroup` (`bifactor_multigroup.py`) no longer
  defaults `max_iter`, `tol`, `n_starts`, or `seed`.** Same reasoning as
  `fit_bifactor_grm`.
- **`run_bifactor_bootstrap` (`bifactor_bootstrap.py`) no longer defaults
  `base_seed`, `ci_level`, `max_iter`, `n_starts`, or `tol`.** `base_seed` is
  a fixed stochastic seed; `ci_level` is an unsourced decision threshold;
  `max_iter`/`tol` are unsourced convergence controls; `n_starts` is an
  unsourced replicate count. These five parameters are now required and
  keyword-only (a `*` was added ahead of them to keep the remaining
  optional parameters — `group_ids`, `n_groups`, `anchor_mask`, `n_jobs`,
  `device`, `estimate_specific_vars` — keyword-only-compatible without
  reordering the required, non-defaulted set ahead of them positionally).
- **`assess_bifactor_scoreability` / `assess_bifactor_scoreability_from_logit_slopes`
  no longer default `zero_tolerance`.** `0.0` was an unsourced
  decision-threshold flag cutoff; it is now a required keyword-only
  argument on both the new names and their deprecated aliases.
- No "keep" decision in this issue's slice of
  `docs/api/renames-and-defaults-20260917.csv` (device, `q_general`,
  `q_specific`, `newton_iter`, `ridge`, `general_factor`, `n_groups`,
  `n_jobs`, `group_ids`, `anchor`/`anchor_mask`,
  `estimate_specific_vars`) carries a cited APA 7th source in its
  `default_rationale` column, so no docstring citations were added for
  this issue; the existing Golub & Welsch (1969) / Gibbons et al. (2007) /
  Cai, Yang & Hansen (2011) / Andrews & Buchinsky (2000) references on
  these five modules are unchanged.
- **Not yet mirrored on the PyO3 side (Python-only change; follow-up
  needed).** The compiled `crates/fast-mlsirm-py` bindings still default
  `max_iter`, `n_starts`, `seed` (and `tol`) on `fit_bifactor_grm` and
  `fit_bifactor_grm_multigroup`, and `max_iter`/`tol` on
  `fit_bifactor_grm_fipc` (`#[pyo3(signature = (...))]` in
  `crates/fast-mlsirm-py/src/lib.rs`). A follow-up PR should remove those
  PyO3-side defaults so the two layers agree.

#### ADR-0028 naming/defaults applied to core IRT fitters (#1964)

- **73 previously-defaulted parameters across `fast_mlsirm.grm`,
  `fast_mlsirm.gpcm`, `fast_mlsirm.twopl`, `fast_mlsirm.two_tier_grm`,
  `fast_mlsirm.nominal`, `fast_mlsirm.rsm`, `fast_mlsirm.mhrm`,
  `fast_mlsirm.rasch_cml`, `fast_mlsirm.lltm`, `fast_mlsirm.mixed`,
  `fast_mlsirm.mixture`, `fast_mlsirm.objective`,
  `fast_mlsirm.estimators.marginal`, and `fast_mlsirm.estimators.mmle` are
  now required caller arguments**, per ADR-0028's defaults policy: node
  counts (`q_theta`, `q_xi`, `q_u`, `xi_points`, `n_nodes`, `nevalpoints`),
  iteration/convergence controls (`max_iter`, `max_cycles`, `burn_in`,
  `mh_steps`, `tol`, `target_accept`, `m_steps`, `eps_distance`),
  replicate/restart counts (`n_starts`), stochastic seeds (`xi_seed`,
  `seed`), and model-family selectors (`model`) on generic fitters. None of
  these had a cited source for its exact numeric default (the ADR-0028
  Context section found zero in-docstring citations for any of them across
  the package), so none qualifies for `keep`.
- Every affected function keeps parameters with a stated `keep` decision
  (e.g. `q`/`node_rule` on the quadrature-based fitters, `estimate_se`,
  `estimate_corr`, `family`, `ridge_a`/`ridge_b`, `latent_dim`, `n_threads`)
  at their existing defaults — those already have a `keep` rationale in
  `docs/api/renames-and-defaults-20260917.csv` outside this change's scope.
- New parameters became keyword-only (`*,` inserted before the first
  newly-required parameter that would otherwise follow a still-defaulted
  one) only where needed to keep the signature syntactically valid;
  existing positional call sites for parameters that stayed positional
  (e.g. `model` on `fit_grm`/`fit_gpcm`/`fit_nominal`/`fit_2pl`) are
  unaffected.
- Every in-package, test, and doc call site was updated to pass the
  previously-implicit default explicitly, so no numerical behavior changes
  — this is a required-argument change only, not a default-*value* change,
  and `tests/test_rust_parity.py` (Rust<->NumPy numerical parity) remains
  green.
- No PyO3/`fast_mlsirm._core` signature required a matching change for
  this issue's functions: the compiled entry points these Python wrappers
  call already take the resolved values as plain positional/keyword
  arguments with no Rust-side default to drop.


### Deprecated

#### ADR-0028 naming/defaults applied to `dif`, `deltaplot`, and `polytomous` (#1962)

- **`fast_mlsirm.dif`: renamed to verb-first names.** `mantel_haenszel_dif`
  -> `detect_dif_mantel_haenszel`, `mantel_haenszel_dif_purified` ->
  `detect_dif_mantel_haenszel_purified`, `logistic_dif` ->
  `detect_dif_logistic`, `logistic_dif_purified` ->
  `detect_dif_logistic_purified`, `mantel_smd_dif` -> `detect_dif_mantel_smd`,
  `gmh_dif` -> `detect_dif_gmh`, `breslow_day_dif` -> `detect_dif_breslow_day`.
  The old names remain as deprecated aliases (identical signature and
  defaults) for one minor release, emitting `DeprecationWarning`, and stay
  exported from the same places as before (`fast_mlsirm/__init__.py` via
  `_legacy_init.py`).
- **`fast_mlsirm.polytomous`: renamed to verb-first names.**
  `bifactor_expected_total_score_monotonicity` ->
  `check_bifactor_expected_total_score_monotonicity`,
  `cat_simulate_polytomous` -> `simulate_cat_polytomous`, `dif_polytomous` ->
  `detect_dif_polytomous`, `dif_polytomous_anchor_sets` ->
  `detect_dif_anchor_sets_polytomous`, `dif_polytomous_purified` ->
  `detect_dif_polytomous_purified`, `expected_total_score_monotonicity` ->
  `check_expected_total_score_monotonicity`,
  `focal_expected_total_score_monotonicity` ->
  `check_focal_expected_total_score_monotonicity`, `information_polytomous`
  -> `compute_information_polytomous`, `item_fit_polytomous` ->
  `compute_item_fit_polytomous`, `local_dependence_polytomous` ->
  `diagnose_local_dependence_polytomous`, `person_fit_polytomous` ->
  `compute_person_fit_polytomous`, `polytomous_category_probabilities` ->
  `predict_category_probabilities_polytomous`, `polytomous_expected_response`
  -> `predict_expected_response_polytomous`, `polytomous_information_criteria`
  -> `compute_information_criteria_polytomous`, `u3_cutoff_polytomous` ->
  `compute_u3_cutoff_polytomous`, `u3_person_fit_polytomous` ->
  `compute_u3_person_fit_polytomous`. Same one-minor-release deprecated-alias
  policy as above.

#### ADR-0028 naming/defaults applied to the bifactor modules (#1963)

- **`fast_mlsirm.bifactor_recursion.bifactor_lord_wingersky` is renamed to
  `enumerate_bifactor_lord_wingersky`.** The old name remains available for
  one minor release and emits `DeprecationWarning`; it forwards to the new
  name with identical behavior.
- **`fast_mlsirm.bifactor_recursion.direct_enumeration_bifactor` is renamed to
  `enumerate_bifactor_direct`.** Same one-minor-release deprecated-alias
  treatment as above.
- **`fast_mlsirm.bifactor_scoreability.bifactor_scoreability` is renamed to
  `assess_bifactor_scoreability`.** Same one-minor-release deprecated-alias
  treatment; both old and new names are exported from `fast_mlsirm` and
  `fast_mlsirm.bifactor_scoreability`.
- **`fast_mlsirm.bifactor_scoreability.bifactor_scoreability_from_logit_slopes`
  is renamed to `assess_bifactor_scoreability_from_logit_slopes`.** Same
  one-minor-release deprecated-alias treatment.

#### ADR-0028 naming/defaults applied to core IRT fitters (#1964)

- **Six renamed callables keep a one-minor-release alias that emits
  `DeprecationWarning`, per ADR-0028's naming convention:**
  - `fast_mlsirm.estimators.marginal.category_logprobs` -> `compute_category_logprobs`
  - `fast_mlsirm.estimators.marginal.gpcm_node_gradient` -> `compute_gpcm_node_gradient`
  - `fast_mlsirm.estimators.marginal.grm_category_logprobs` -> `compute_grm_category_logprobs`
  - `fast_mlsirm.ksirt.ksirt_analysis` -> `analyze_ksirt` (also re-exported,
    old and new, from `fast_mlsirm/__init__.py` and `_legacy_init.py`, same
    as before)
  - `fast_mlsirm.objective.linear_predictor` -> `compute_linear_predictor`
  - `fast_mlsirm.objective.model_flags` -> `get_model_flags`
  Each old name is now a thin wrapper that warns and delegates to the new
  name with identical behavior; the alias and its `_legacy_init.py` entry
  are deleted at the start of the next minor release.

### Fixed

#### Bifactor GRM dense-quadrature EM stall (#1976 / #1981)

- Skip Gauss–Hermite nodes whose prior weight underflows to exact zero in the
  bifactor GRM E-step (CPU and GPU) so `gen_log - log_wg` no longer forms
  `(-inf) - (-inf)` NaNs that poison expected counts and freeze the M-step at
  the start slopes.
- When relative loglik change would claim `tolerance_met` but every item
  parameter is still bit-identical to the start, report `converged=False` /
  `termination_reason="numerical_em_stall"` instead of a false success.
- Document the new termination reason on single-group and multigroup bifactor
  fit result surfaces; pin the contract with Rust regression tests (including
  a dense q=421 fit) and a Python issue-repro gate.

## [0.11.0] - 2026-09-17

### Added

#### Bifactor GRM observed-information standard errors (Oakes)

Stage-3 standard errors for the single-group polytomous bifactor graded
response model (stage 3 of #1912): `mlsirm_core::bifactor_oakes::
bifactor_oakes_se` (with `BifactorOakesConfig`) returns the full
item-parameter observed information, its inverse vcov, and standard errors
via the Oakes (1999, eq. 6) identity, evaluated at given item parameters.
The complete-data gradient and Hessian are analytic
(`poly::grm_node_hessian`, cross-checked against central finite differences
in tests); the cross term re-runs the Gibbons-Hedeker reduced E-step once
per free parameter. The assembly is written behind a `PosteriorProvider`
trait so the stage-2 multigroup calibration reuses it with group-specific
E-steps. A non-positive-definite information matrix is reported with
`positive_definite = false` and a `non_pd_reason`, while `vcov`/`se` are
`None` — never a generalized inverse or any other substitute (#1912
acceptance criterion 3). Python surface:
`fast_mlsirm.bifactor_grm.bifactor_oakes_se` returning `BifactorOakesSe`
(quadrature counts and the cross-term step are required caller arguments).
Evidence: analytic-vs-finite-difference Q-Hessian agreement; Oakes
information vs numerical Hessian of the exact marginal log-likelihood on a
tiny problem; mean-SE vs empirical-SD agreement over 100 simulation
replicates at recovery scale (worst 0.235, Monte Carlo noise ~7%);
small-grid convergence stabilization; mirt `SE.type = "Oakes"`
agreement on the committed fixture (SEs 8.4e-3, vcov 1.9e-2 at matched
quadpts = 15); study-settings grid convergence at 121 vs. 241 Gauss-Hermite
nodes per dimension (#1929's node-count cap removal made this grid
reachable) — Oakes SEs agree to `maxRel|dSE| = 8.76e-9`, run once locally
in `bifactor_oakes_calibration::study_settings_se_converges_at_121_vs_241_nodes`
(`#[ignore]`d as long-running, ~46 min at `q=241`).

#### GPU-parallel bifactor E-step and joint person bootstrap with caller-controlled stopping

- Add a GPU-parallel E-step for the Bock-Aitkin bifactor GRM with
  Gibbons-Hedeker dimension reduction
  (`crates/mlsirm-core/src/gpu_bifactor.rs`), covering the single-group
  estimator and the multigroup calibration (including group moment
  accumulators). Kernels accumulate in f32; CPU/GPU fit-level agreement is
  asserted within a documented single-precision tolerance on fixtures
  including reverse-keyed items and multiple groups. CPU fallback when no
  GPU adapter is available; `device` is a validated argument
  (`cpu`/`gpu`/`auto`) on both configs, both PyO3 entry points, and both
  Python wrappers.
- Release the GIL around the Rust bifactor fits (`py.detach`) so the
  bootstrap thread pool parallelizes.
- Add a joint person bootstrap driver
  (`fast_mlsirm.bifactor_bootstrap.run_bifactor_bootstrap`) in which
  replicate count, batch size, Monte Carlo stopping ratio, and compute
  budget are caller arguments with validated ranges. The stopping rule is a
  sequential application of the endpoint-accuracy framework of Andrews and
  Buchinsky (2000, §§ 2–4): the run stops once the maximum
  percentile-interval endpoint movement relative to the interval half-width
  falls below the caller ratio. Per-replicate convergence is reported;
  failed replicates are excluded, never substituted.
- Report bootstrap percentile intervals alongside empirical standard errors.
- Add two-stage Lord-Wingersky score recursion
  (`fast_mlsirm.bifactor_recursion`) matching direct enumeration within
  1e-12.
- Assert stage-5 CPU/GPU fit-level parity at the maintainer-standard
  study-precision quadrature grids (`tests/test_bifactor_gpu_high_q.py`,
  gated behind `STAGE5_HIGH_Q=1` like the Rust `#[ignore]` node-count
  regressions): CPU and GPU E-steps agree within the documented
  single-precision envelope at 121 and 241 nodes per dimension, and the
  CPU fits at 121 vs 241 nodes agree within the 5e-3 numerical band
  (marginal-likelihood integral convergence). Quadrature counts are caller
  arguments with no defaults and no caps: any `n >= 1` resolves via the
  shared arbitrary-`n` Gauss-Hermite rule
  (`quadrature::require_gh_rule`, Golub & Welsch, 1969; #1929/#1945),
  replacing the fixed `SUPPORTED_Q` membership table this branch
  previously enforced.
- Measure the joint person bootstrap at the 121-point study grid
  (`test_joint_bootstrap_cpu_vs_gpu_wall_time_q121`, same gate):
  measured CPU vs GPU wall times are printed for the PR record rather
  than asserted against machine-specific thresholds.
- Measured study-grid evidence (Apple Silicon, `STAGE5_HIGH_Q=1`, tiny
  48-person/6-item/2-specific fixture, `tol=1e-3`): single-group parity
  at `q=121` — CPU 4.544s vs GPU 6.181s, `max|Δslope|=1.897e-07`,
  `max|Δthreshold|=1.138e-07`, `|Δloglik|=2.374e-05` (4 EM iterations,
  both converged); at `q=241` — CPU 21.642s vs GPU 26.561s,
  `max|Δslope|=1.326e-07`, `max|Δthreshold|=1.356e-07`,
  `|Δloglik|=3.232e-05` (4 iterations, both converged); CPU 121-vs-241
  agreement `|Δloglik|=4.829e-09` (integral converged). Joint bootstrap
  at `q=121` (`B=2`, two-group multigroup path): CPU 161.512s
  (80.756s/rep) vs GPU 210.261s (105.130s/rep), replicate-by-replicate
  parity within 1e-3. Small problems stay CPU-faster (per-sweep GPU
  buffer setup dominates), as already disclosed in ADR-0027.

#### Single-group polytomous two-tier GRM with reduction over the specific tier (stage 4 of #1912)

- Add a single-group full-information polytomous two-tier graded response
  fitter (Cai, 2010; Cai, Yang, & Hansen, 2011, eq. 6-7; Gibbons et al.,
  2007, eq. 9/15): caller-supplied confirmatory primary pattern with an
  estimated primary correlation matrix, at most one orthogonal specific
  factor per item, unconstrained slopes, strictly decreasing boundary
  intercepts, Bock-Aitkin EM integrating only `P + 1` dimensions
  (fixed-grid primaries with Phi reweighting, per-block specific sums),
  deterministic multi-start selection, primary-factor EAP scores with
  posterior SDs, and a `fit_two_tier_grm` Python binding. Reduces exactly to
  the stage-1 bifactor GRM at one primary dimension. Validated against
  `mirt::bfactor` with a two-tier specification on a committed fixture
  (slopes/intercepts/correlation/loglik agreement bands with measured values
  reported in the tests).

#### Public API naming convention and unsourced-defaults policy (#1959)

- **ADR-0028** (`docs/adr/0028-public-api-naming-and-defaults-policy.md`,
  Proposed): one verb-first naming convention and one unsourced-defaults
  policy (numerical-precision controls, decision thresholds, seeds, model
  choice) for every public `fast_mlsirm` callable and PyO3 entry point.
- `tools/inventory_public_api.py`: regenerates a full public-callable
  inventory (`docs/api/inventory-YYYYMMDD.csv`) via static analysis, no Rust
  build required.
- `tools/classify_renames_and_defaults.py`: mechanically applies ADR-0028's
  rules to the inventory, producing
  `docs/api/renames-and-defaults-YYYYMMDD.csv` with a proposed name and a
  per-default decision (`keep+source` / `require` / `change`) for every
  callable, absorbing #1958/#1960's completed `dif_polytomous*` outcome.
No code, name, or default changes in this PR (Phase 1, documentation only);
per-module implementation is tracked in the sub-issues this PR opens.

#### Rust-owned OLS with HC0–HC3 sandwich covariance (#1982)

- Expose Rust-owned ordinary least squares with MacKinnon–White HC0–HC3
  heteroskedasticity-consistent covariance through `fast_mlsirm.fit_ols_hc`,
  plus `contrast` (Wald χ²(1) with t/F tails) and upper-tail helpers
  `chi2_sf_df1`, `f_sf`, and `t_sf`. Numerical work stays in `mlsirm-core`
  (`regression`); the Python module only validates NumPy layout and marshals
  results — no SciPy or Rscript dependency on this path (MacKinnon & White,
  1985; Long & Ervin, 2000).
- Bound design size (`n ≤ 1_000_000`, `k ≤ 1_024`), require finite float64
  evidence, and require residual `df = n - k` for contrast t/F tails.
- Rust integration tests (`crates/mlsirm-core/tests/regression_ols.rs`) and
  Python Rust↔parity gates (`tests/test_rust_regression_ols_hc_parity.py`).

#### Fixed-item parameter calibration (FIPC) for polytomous GRM

- Add fixed-item parameter calibration for the unidimensional GRM
  (`mlsirm_core::poly::fit_poly_fipc`, Python `fast_mlsirm.polytomous.fit_poly_fipc`)
  and the polytomous bifactor GRM
  (`mlsirm_core::bifactor_grm::fit_bifactor_grm_fipc`, Python
  `fast_mlsirm.bifactor_grm.fit_bifactor_grm_fipc`): caller-fixed anchor
  items from a reference calibration plus a flag vector; non-anchor item
  parameters and the focal population's latent mean/variance (general
  factor; specific variances where identified) estimated by MML-EM with the
  prior distribution updated after every M-step — the MWU-MEM method (Kim,
  2006, eqs. 14-15, pp. 361-362), the only compared variant that recovered
  shifted focal distributions without under-estimation (pp. 377-378; Paek &
  Young, 2005). No rescaling of the latent points after an EM cycle, no
  reflection canonicalization: fixed anchors pin the orientation, including
  reverse-keyed anchors. Quadrature node counts stay caller arguments; the
  unidimensional rule set gains the 121-node Gauss-Hermite rule
  (`numpy.polynomial.hermite_e.hermegauss(121)`, weights normalized) as the
  study floor.
- Validate against `mirt::fixedCalib` (MWU-MEM default) on a committed
  unidimensional GRM fixture (`tests/fixtures/poly_fipc_grm/`, R 4.x with
  mirt 1.46.1): free-item agreement plus a same-objective profile-likelihood
  corroboration (mirt's stacked-data empirical-histogram loglik is recorded
  but not directly comparable). Recovery under a known mean/variance shift,
  equivalence to concurrent calibration with anchors fixed at truth, and
  reverse-keyed anchors are covered for both models; study-scale runs
  (`N = 1,020`, `q_theta = 121` unidimensional) run as ignored
  statistical-studies tests with bands from measured multi-seed spreads.
  Paper basis: Kim (2006, JEM 43(4), 355-381,
  https://doi.org/10.1111/j.1745-3984.2006.00021.x) and Paek & Young (2005,
  AME 18(2), 199-215, https://doi.org/10.1207/s15324818ame1802_4).

### Changed

#### Clippy lint triage for #1905

- Documented as intentionally left: `needless_range_loop` (114, all
  `HasPlaceholders`, each needs per-site judgment in numeric kernels),
  `too_many_arguments` (28, public API shape; repo convention is
  per-fn allow), `neg_cmp_op_on_partial_ord` (14, load-bearing NaN
  guards where the lint suggestion would change validation behavior),
  `type_complexity` (5, public signatures), `if_same_then_else`
  (2, intentional degenerate arms with distinct documented reasons).
  Test-target warnings are triaged as follow-up in the issue.

#### GPU-parallel bifactor E-step and joint person bootstrap with caller-controlled stopping

- `q_general`/`q_specific` validation in the stage-5 Python surface
  (`bifactor_bootstrap.run_bifactor_bootstrap`) now accepts any integer
  `n >= 1` instead of the removed fixed-table membership set, matching
  the merged estimator contract (#1929/#1945); out-of-range counts still
  fail loudly and are never clamped.

#### Single-group polytomous two-tier GRM with reduction over the specific tier (stage 4 of #1912)

- `q_primary`/`q_specific` validation now resolves the shared arbitrary-`n`
  Gauss-Hermite quadrature (`quadrature::require_gh_rule`, any `n >= 1`,
  #1929) instead of a fixed `SUPPORTED_Q` membership table, matching the
  removal of that table's cap. Add a `#[ignore]`d 121-vs-241 node-count
  numerical-agreement regression (`two_tier_grm_node_agreement.rs`,
  single-primary/single-specific design to keep the primary product grid
  tractable), executed locally (`cargo test --release -- --ignored
  --nocapture`) at both node counts: `q=121` converged in 6 iterations
  (2.20s, final loglik -1246.539916) and `q=241` converged in 6 iterations
  (8.12s, final loglik -1246.539916) — `|loglik diff| = 0.000000`, well
  inside the 5e-3 tolerance.

#### Verify Bock & Zimowski (1997) locators in bifactor/multigroup docs (#1927)

- **Full-text verification attempted, chapter unobtainable.** Bock &
  Zimowski (1997), *Multiple group IRT* (Handbook of Modern IRT, ch. 25,
  pp. 433-448), cited in `crates/mlsirm-core/src/bifactor_grm.rs` and
  `poly::fit_poly_multigroup`, is absent from the maintainer's Zotero
  library and local paper cache, has no open-access copy, and the Springer
  chapter page redirects to an institutional login; the KW library
  (kupis.kw.ac.kr) document-delivery/e-book route could not be completed
  because Chrome browser automation was unavailable this session. Only the
  publisher's own chapter metadata (chapter 25, pp. 433-448) was
  independently confirmed via Springer's DOI record and WorldCat.
- **No internal locator needed correcting.** Both citation sites already
  claimed no chapter-internal equation or page locator (the pooling claim
  was labeled "conceptual"), so there was nothing unverifiable to remove.
  Both comments now record the verification attempt and its outcome, and
  point to the already page-verified Cai, Yang, & Hansen (2011, Zotero
  `TNQ22C7T`) and Bock & Aitkin (1981) references — read in full — as the
  independently verified sources for the same reference-group multigroup
  pooling this chapter describes.

#### Unsourced defaults removed from the polytomous DIF entry points (#1958)

- **`dif_polytomous`, `dif_polytomous_purified`, and
  `dif_polytomous_anchor_sets` no longer default `model`, `q_theta`,
  `max_iter`, `tol`, `fdr_q`, `max_rounds` (the latter two functions), or
  `min_anchor_items`.** All are now required caller arguments. Previously
  `model` silently defaulted to `"gpcm"` (differing from the `"grm"` used
  elsewhere in this package's own study measurement models) and `q_theta`
  defaulted to `21`, a Gauss-Hermite node count with no accuracy target on
  file to source it against — the exact violation of the #1929 quadrature
  rule (node counts are caller arguments, no defaulted value below the
  project's 121-node floor) that this issue reports. `max_iter`, `tol`,
  `fdr_q`, `max_rounds`, and `min_anchor_items` have the same problem: none
  of `200`, `1e-5`, `0.05`, `3`, or `4` has a documented source in this
  repository, and this package does not ship a default it cannot defend.
- **Same reasoning already applied on this file.**
  `focal_expected_total_score_monotonicity` and
  `bifactor_expected_total_score_monotonicity` already require `q_nuisance`
  / `q_specific` with no default under #1929; this change extends that
  requirement to the sibling tuning constants on the three polytomous DIF
  functions rather than leaving them as a special case.
- **`fdr_q` and the purification loop have citable conventions, even though
  neither is defaulted.** `0.05` is the illustrative FDR level used
  throughout Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
  discovery rate: A practical and powerful approach to multiple testing.
  *Journal of the Royal Statistical Society: Series B (Methodological),
  57*(1), 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x — and
  the anchor-rebuild-and-repeat purification loop itself is Candell, G. L.,
  & Drasgow, F. (1988). An iterative procedure for linking metrics and
  assessing item bias in item response theory. *Applied Psychological
  Measurement, 12*(3), 253-260.
  https://doi.org/10.1177/014662168801200304 — but neither source states a
  specific round count, so `max_rounds` still has no defensible default and
  stays required.
- **Breaking change, audited against the codebase's other DIF/polytomous
  entry points.** The observed-score DIF functions in `dif.py`
  (`mantel_haenszel_dif`, `mantel_haenszel_dif_purified`,
  `logistic_dif`, `logistic_dif_purified`, `sibtest`, `mantel_smd_dif`,
  `gmh_dif`, `breslow_day_dif`) and the other polytomous fit/diagnostic
  functions in `polytomous.py` were checked for the same pattern: none of
  them defaults a Gauss-Hermite node count (they either take none, or -- for
  the already-fixed `focal_expected_total_score_monotonicity` /
  `bifactor_expected_total_score_monotonicity` -- already require it), so
  they are out of scope for this issue's #1929 violation. Their `fdr_q`
  defaults are unchanged.

#### Fixed-item parameter calibration (FIPC) for polytomous GRM

- `BifactorFipcConfig.q_general`/`q_specific` validation (and the Python
  `fit_bifactor_grm_fipc` wrapper) now resolve the shared arbitrary-`n`
  Gauss-Hermite quadrature (`quadrature::require_gh_rule`, any `n >= 1`,
  #1929) instead of the removed fixed `SUPPORTED_Q` table, matching
  `fit_bifactor_grm`/`fit_bifactor_grm_multigroup`. Move the ignored
  `bifactor_fipc_study_n1020` study test from `q_general=31..41,
  q_specific=21..31` to the maintainer's >= 121-node-per-dimension floor
  (`q_general = q_specific = 121` for both the reference and FIPC fits);
  executed locally in release mode (668s), converged, all recovery
  assertions passing.
- `quadrature::require_gh_rule_unidim` now actually dispatches through the
  embedded 121-node unidimensional table (`gh_rule_121`,
  `numpy.polynomial.hermite_e.hermegauss(121)`) instead of silently falling
  back to the generic arbitrary-`n` path at every node count — the wiring
  bug that made the embedded table dead code is fixed so `fit_poly_fipc`'s
  `q_theta = 121` study path uses it as intended; re-verified locally
  (`fipc_study_recovery_n1020_q121`, release mode, 7.14s, passing).

### Fixed

#### Fail-closed pytest outcomes

- Escalate any pytest invocation with a skip, import-or-skip, skipif, xfail, or xpass outcome to a non-zero exit status via a session-level enforcement plugin, so a successful suite proves every collected evidence lane executed.
- Count collection-time skips as well as setup/call/teardown skips, and fail closed when outcome accounting cannot be observed.
- Review capability-gated non-executions through an explicit allowlist instead of rewriting capability-specific tests.
- Treat unexpected passes as failures by default with strict xfail handling.

#### Graphify tooling investigation for #1847 and #1833

- #1847: `to_json`'s node-count shrink guard refused a `cluster-only` write
  on an unchanged graph after `build_from_json`'s ghost-merge pass
  legitimately collapsed a manifest-derived duplicate node into its
  AST-canonical twin (`crate:mlsirm-core` / `pkg_mlsirm_core`, zero
  incident edges dropped). `build_from_json` now records the collapsed
  count (`_ghost_dedup_count`); the shrink guard excuses a drop only when
  fully explained by it. Upstream PR:
  https://github.com/Graphify-Labs/graphify/pull/3623.
- #1833: a workspace-only `Cargo.toml` (`[workspace]`, no `[package]`)
  correctly emits no package node, but the extractor's zero-node detector
  could not tell that apart from an unexplained failure and printed a
  persistent warning every run. The manifest parser now marks this case
  `skipped`, so the by-design exclusion is explicit instead of warning.
  Upstream PR: https://github.com/Graphify-Labs/graphify/pull/3622.
- No fast-mlsirm runtime, Cargo, or Python code changed — both issues were
  tooling-only (Graphify artifact refresh/reviewability), confirmed via a
  RED-then-GREEN regression test in the `seonghobae/graphify` fork before
  the upstream PRs were opened.
- Pinned install/rollback instructions for trying the fork fix locally are
  recorded on fast-mlsirm#1833.

#### Clippy lint triage for #1905

- Resolve the 24 deny-level `clippy::erasing_op` errors: every site was
  verified (twice, independently) to be the intentional flat row-major
  index idiom (`0 * stride` keeps rows aligned), with no genuine bug.
  Narrow function-level `#[allow]`s with a one-line reason were added;
  no crate-wide allow. `cargo clippy --workspace --all-targets` exits 0.
- Apply clearly-safe machine lints with no behavior change
  (`map_or` to `is_none_or`/`is_some_and`, `contains`, range-contains,
  `is_multiple_of`, `div_ceil`, needless borrows, `copy_from_slice`,
  NaN-preserving boolean simplification, single-match/collapsible/
  for-values/obfuscated-if rewrites, unnecessary cast, doc-list
  indentation). Lib warnings 380 -> 279.
- Truncate non-representable float digits (`excessive_precision`) in
  quadrature tables and numeric constants. All 116 changed literals
  parse to bit-identical `f64` values (verified programmatically);
  the quadrature tables' shortest-roundtrip claim now holds.
  Lib warnings 279 -> 163.

#### GPU-parallel bifactor E-step and joint person bootstrap with caller-controlled stopping

- Remove the crate-wide `clippy::erasing_op` / `clippy::identity_op`
  allowance; no broad lint suppression remains.

#### `logistic_dif_purified` purifies again, on `flagged_bh` (#1941)

- **`logistic_dif_purified` no longer no-ops.** Its anchor-purification
  criterion was `purify_flagged(jg_class)` (`jg_class in {B, C}`). #1880
  retired `jg_class` to `"U"` ("not applicable") for every item, so that
  criterion could never fire: `n_anchor` stayed at the initial item count and
  `rounds` stayed `0` regardless of the DIF actually present in the data.
  This silently downgraded the function to an expensive wrapper around
  `logistic_dif` that never purified anything.
- **Replacement criterion: `flagged_bh`.** The purification loop now drops an
  item from the anchor when its Benjamini-Hochberg-adjusted `chi2_total`
  omnibus test (`flagged_bh`) rejects — the same multiplicity-controlled
  significance test `dif_polytomous_purified` uses for the identical reason:
  no calibrated practical-significance class (an ETS-style B/C letter)
  exists for this statistic. Jodoin and Gierl's (2001) `.035`/`.070` bands
  are stated on a Zumbo-Thomas weighted-least-squares one-degree-of-freedom
  partition this package does not compute, not on the two-degree-of-freedom
  Nagelkerke pseudo-R² `delta_r2` this package reports (see #1880's
  fragment), so `flagged_bh` is used directly rather than guessing a class.
  This makes the loop's anchor MORE aggressive at large `N` than
  `mantel_haenszel_dif_purified`'s practical-significance screen, not less —
  documented on the function.
- **No public API change.** `logistic_dif_purified`'s signature and return
  keys (`anchor`, `n_anchor`, `rounds`, `purify_converged`,
  `purify_termination_reason`, plus every `logistic_dif` key) are unchanged.
  Only the purification behavior — which items the anchor excludes, for data
  with DIF present — changes, from "never" to "on `flagged_bh`".
  `jg_class` itself is untouched and remains `"U"` for every item.
- **Caveat inherited, not introduced.** As before, the anchor is selected
  from the same data it is then tested against, so the returned p-values are
  conditional on a data-dependent selection and Benjamini-Hochberg does not
  carry an FDR guarantee for the purified sweep; treat `flagged_bh` as a
  screening device (see the function's existing docstring caveats, unchanged
  by this fix).
- **References.** Candell, G. L., & Drasgow, F. (1988). An iterative
  procedure for linking metrics and assessing item bias in item response
  theory. *Applied Psychological Measurement, 12*(3), 253-260.
  https://doi.org/10.1177/014662168801200304 — Zumbo, B. D. (1999). *A
  handbook on the theory and methods of differential item functioning
  (DIF): Logistic regression modeling as a unitary framework for binary and
  Likert-type (ordinal) item scores* (p. 27). Directorate of Human Resources
  Research and Evaluation, Department of National Defense.

#### Stage-1 bifactor GRM mirt fixture regenerated with corrected category order (#1950)

- `tests/fixtures/bifactor_grm_stage1/generate_mirt_fixture.R` compared the
  simulated uniform draw against each graded-response boundary probability
  with `u > p_k`, which reversed the intended category order (higher latent
  trait produced *lower* observed categories). The nested-event identity
  `P(Y >= k) = P(u < p_k)` requires `u < p_k`; regenerated `dataset.csv` and
  `mirt_fixture.json` with the corrected rule and added a category-order
  guard (`cor(theta_g, rowSums(resp)) > 0.3`) so a reversed rule fails fast
  next time instead of only being caught by inspection.
- `crates/mlsirm-core/tests/bifactor_grm_mirt_agreement.rs` still passes
  unchanged: the Rust<->mirt comparison canonicalizes reflection per
  dimension before comparing slopes/intercepts/log-likelihood, so it was
  insensitive to the category-order bug and remains a valid agreement check
  on the corrected fixture.

## [0.10.0] - 2026-09-17

### Added

#### Preregistered external-validation evidence profiles

- Added a domain-neutral, source-text-free `fast_mlsirm.validation_profile` contract for preregistered external evidence. Profiles preserve technical, construct, transportability, fairness, and decision-utility evidence as distinct classes with explicit `passed`, `failed`, `indeterminate`, `not_executed`, and `not_applicable` states; bind exact assessment/rubric/item-bank/model/protocol provenance; reject future-available evidence beyond the analysis cutoff; normalize callback-free fixed-offset timestamps to UTC; bound evidence and limitation collections before iteration; replay profile and nested-evidence invariants before granting public serialization or fingerprint authority after post-construction mutation; and expose a deterministic SHA-256 profile fingerprint without aggregating away failed or unavailable evidence. The slice performs validation, serialization, and provenance only; future statistical validity, transportability, fairness, and utility arithmetic remains Rust-owned.

#### Finite-population proportion sampling design

- Added a domain-neutral Rust/PyO3 `fast-mlsirm.sampling-design.v1` contract for normal-approximation finite-population proportion sample size, finite-population correction, and caller-selected proportional or equal-cost Neyman stratum allocation. Python only validates and marshals exact caller evidence; sample-size, correction, and allocation arithmetic remains Rust-owned.
- The immutable result retains canonical ordered inputs and binds Rust-generated source/input/output SHA-256 identities to the stable source identity and algorithm version. Callers keep sample-frame and selected-membership provenance outside the arithmetic artifact.

#### Pin signed-slope recovery for reverse-keyed graded items (#1870)

- Add contract coverage confirming `fit_grm` recovers negative (reverse-keyed) slopes with no non-negativity bound: simulating graded data from the package-native form with known negative slopes and refitting recovers them (max absolute error 0.10 across three reverse-keyed items in the fixture), with zero exact-zero estimates. No product code changed; `fit_grm`'s unconstrained-slope contract was previously documented but untested against a true negative slope.

#### Report when a fit_mixed_items estimate rests on an optimizer bound (#1882, refs #1881)

- Add `MixedItemEstimate::at_bound` (surfaced through the PyO3 binding as `MixedItemParameters.at_bound`), listing which parameter roles (`"slope"`, `"latent_position"`, `"parameter"`) are held at one of `clamp_params`'s three bounds (`[-12, 12]` on every free parameter, `[-5, 4]` on a free-slope family's working `log a`, `[-6, 6]` per latent coordinate of a spatial family) at the returned solution. Empty is the normal case. Previously an estimate resting on a bound (e.g. a reverse-keyed item's slope pushed onto the `exp(-5)` positivity floor and reported as `0.006738`, indistinguishable from a genuinely low-discrimination item) was reported with no indication it was a boundary value rather than an interior optimum.

#### Pin which reflection each item family absorbs (#1883, refs #1881)

- Add contract coverage classifying which `(a, theta)` or `(theta, delta)` reflection each `fit_mixed_items` item family absorbs, checked against the model cells directly rather than existing helper predicates: `TwoPl`/`Grm`/`Gpcm`/`Sequential`/`Lsirm*` absorb a joint `(a, theta)` reflection (slope anchor applies); `Rasch`/`Cll`/`Tutz` absorb none and pin the orientation themselves; `Ideal`/`Ggum` absorb a joint `(theta, delta)` reflection with the slope untouched, so no slope anchor can pin them. No product code or behavior changed.

#### Expected-total-score monotonicity diagnostic (#1888, closes the unidimensional half of #1873)

- Add a unidimensional expected-total-score monotonicity diagnostic built on the existing `polytomous_expected_response` curve: rather than a single boolean, it reports the theta interval(s) of decrease and the total decrease (the integral of the negative part of the derivative), both of which are stable under grid refinement, unlike a raw count or maximum decrease.

#### Focal-dimension monotonicity for multidimensional graded fits (#1889, closes #1873)

- Add `focal_expected_total_score_monotonicity`, extending the #1888 diagnostic to compensatory multidimensional graded fits: each item's nuisance dimensions marginalize to a single one-dimensional Gaussian integral under the fitted `theta ~ MVN(0, I)` prior, so the focal-dimension curve reduces to the existing unidimensional numeric path with no new multidimensional kernel.
- `q_nuisance` is a required caller argument (no unsourced default quadrature-node count).

#### Purified polytomous DIF sweep with an anchor-eligible set (#1890, addresses requirement 1 of #1874)

- Add `dif_polytomous_purified`: iteratively purifies the polytomous DIF anchor by testing each item against `anchor UNION {itself}` until the flagged set stabilizes, the anchor would fall below `min_anchor_items`, or `max_rounds` is reached — the same rule the dichotomous purified functions use, extended to `dif_polytomous`, which previously tested every studied item against all others with no anchor set returned.
- Returns everything the plain `dif_polytomous` sweep returns, plus `anchor`, `n_anchor`, `rounds`, `purify_converged`, and `purify_termination_reason`, matching the dichotomous purified functions' contract field for field. No Rust-side numeric change; the existing two-group entry point is reused on the restricted anchor column subset.

#### Per-focal-group anchor sets and their intersection (#1891, addresses requirement 3 of #1874)

- Add `dif_polytomous_anchor_sets`: purifies each focal group against the reference on its own two-group subset (via `dif_polytomous_purified`, #1890) and returns the per-group reports, the `n_focal x n_items` anchor matrix, and the intersection anchor a fixed-item calibration can defend for every group at once. Caller-supplied group labels are densified internally and returned unchanged.
- If any group's purification loop ends on `insufficient_anchor_items` or exhausts `max_rounds`, `intersection_trustworthy` is `False` and `untrustworthy_groups` names the affected groups, so a failed purification cannot silently narrow the intersection into a false-conservative result.

#### Single-group polytomous bifactor GRM with Gibbons-Hedeker reduction (stage 1 of #1912)

- Add a single-group full-information polytomous bifactor graded response
  fitter (Gibbons et al., 2007; Gibbons & Hedeker, 1992; Samejima, 1969):
  unconstrained general/specific slopes, strictly decreasing boundary
  intercepts, caller-supplied item-to-specific map with general-only items,
  Bock-Aitkin EM with Gibbons-Hedeker dimension reduction, deterministic
  multi-start selection, general-factor EAP scores with posterior SDs, and a
  `fit_bifactor_grm` Python binding. Unobserved categories fail loudly;
  `max_iter` exhaustion reports `converged=False` instead of substituting
  values. Validated against `mirt::bfactor(itemtype="graded")` on a committed
  fixture (loglik gap 0.015, slope gap <= 0.046, intercept gap <= 0.028).

#### Multiple-group concurrent calibration for the polytomous bifactor GRM (stage 2 of #1912)

- Add `fit_bifactor_grm_multigroup` (Rust `mlsirm_core::bifactor_grm`, Python `fast_mlsirm.bifactor_multigroup`): concurrent multi-group calibration for the bifactor GRM, sharing item parameters across groups with optional per-item anchored/free flags (at least one anchored item required for 2+ groups to link scales).
- Reference group 0 is pinned to N(0, I); focal general-factor mean/variance are estimated, plus optional focal specific-factor variances (means fixed at 0) via `estimate_specific_vars`. Marginal ML uses Bock-Aitkin EM with the Gibbons-Hedeker reduction applied per group; failures are loud per-start errors, never silently clamped.
- Returns per-group EAPs/SDs on the common (reference) scale, per-group category counts, loglik trace, and a convergence flag. Joint cross-group reflection canonicalization reads sign from anchored linking items only. `n_groups == 1` delegates to and bit-reproduces stage-1 `fit_bifactor_grm` (#1925).

#### Bifactor GRM adapter for focal expected-score monotonicity (#1928, links #1873, #1912)

- Add `bifactor_expected_total_score_monotonicity(fit, theta, q_specific)`, an adapter over `focal_expected_total_score_monotonicity` (#1889) for `BifactorGrmFit`/`fit_bifactor_grm` (#1925) with the general factor as the fixed focal dimension; each item's own specific factor is integrated out by caller-sized Gauss-Hermite quadrature bounded by the existing `MAX_POLY_QUADRATURE_POINTS`.
- `q_specific` is a required caller argument (no unsourced default quadrature-node count). Exported from `fast_mlsirm` alongside the existing monotonicity diagnostics.

#### Achieved finite-population proportion

- Added a Rust-owned terminal SRSWOR proportion artifact with design variance,
  Wang/Konijn exact confidence limits, exhaustive coverage tests, and bound
  sampling-design provenance.

#### Binary response state contract

- Add the versioned `fast_mlsirm_binary_response/v1` Measurement contract so dichotomous 0/1 values remain separate from missing, not-observed, abstained, invalid, omitted, not-applicable, insufficient-evidence, and adjudicated states; keep polytomous rubric/facet contracts separate and fail closed instead of thresholding.

#### Lineage channel weight evidence

- Add a Rust fail-closed evidence contract for continuous lineage channel scores
  and an independently accepted criterion anchor. Weight estimation remains
  unavailable until pair-level independent criterion observations are supplied.

### Changed

#### Share one judge-result IRT projection core

- Remove the duplicate category/binning implementation from the explicit criterion-order adapter. `LLMJudgeResult.to_irt_row` remains the single package-owned projection authority; the explicit-order adapter preserves its stricter sealed-mapping checks, delegates once to that canonical projection, and only permutes the validated row into caller-supplied criterion order.
- Add parity regressions for score-derived and explicit-category projections, custom criterion order, delegation to the canonical core, and the sealed result-mapping boundary. No judge scoring threshold, category arithmetic, psychometric likelihood, or Rust numerical behavior changes.

#### Item parameter provenance

- Governed item-bank parameter provenance now distinguishes `provisional` cold-start artifacts from empirically `calibrated` artifacts with immutable, content-addressed evidence. Provisional records must name one bounded domain-neutral method (`rasch_common_discrimination`, `template_prior`, `lltm_predicted`, or `constrained_prior`) and an exact basis fingerprint and cannot carry calibration evidence; calibrated records require an exact calibration-evidence fingerprint and cannot carry provisional provenance. The contract binds item/version, response-model identity, and parameter-artifact fingerprint while storing no raw responses, prompts, provider output, or fitting arithmetic. Item-bank JSON/HTML reports can include this explicit parameter evidence, report `not_supplied` rather than inferring calibration when it is absent, and fail closed on item/version or lifecycle/status mismatches. All parameter estimation, calibration, linking, scoring, uncertainty, and recovery arithmetic remains Rust-owned.

#### `logistic_dif`'s `jg_class` is retired to "not applicable" (#1880)

- **Breaking, for any caller reading `jg_class`.** `logistic_dif` (and
  `logistic_dif_purified`) no longer letter items `"A"`/`"B"`/`"C"` against the
  Jodoin and Gierl (2001) effect-size bands. `jg_class` now reports `"U"`
  ("not applicable") unconditionally, for every item, regardless of fit
  success or omnibus significance. `delta_r2` (the Nagelkerke `R2(M2) -
  R2(M0)` change across the 2-df omnibus) and `delta_r2_uniform` (`R2(M1) -
  R2(M0)`) are unchanged and remain reported as descriptive numbers; neither
  carries a letter class. Callers who branched on `jg_class == "A"`/`"B"`/`"C"`
  must switch to reading `delta_r2` / `delta_r2_uniform` directly, or to
  `flagged_bh` for a significance-only decision.
- `logistic_dif_purified`'s anchor no longer shrinks: its purification
  criterion (`purify_flagged`) only fires on `jg_class in {B, C}`, which is
  now unreachable. This is deliberate, not a silent regression: the obvious
  substitute, raw `flagged_bh` significance, is exactly the over-powered
  test this package's purification design exists to screen against (see the
  doc comment on `logistic_dif_purified`), so substituting it would purify
  far more aggressively than the retired letter class ever did — a
  materially different and worse behaviour change than simply not
  purifying. Callers who relied on `logistic_dif_purified` actually
  shrinking the anchor should track #1880 for a principled replacement
  criterion, or use `mantel_haenszel_dif_purified` (unaffected; it purifies
  on its own ETS `ets_class`, not `jg_class`).
- **Why.** Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I error and
  power rates using an effect size measure with the logistic regression
  procedure for DIF detection. *Applied Measurement in Education, 14*(4),
  329-349, calibrates its `.035`/`.070` bands (p. 335) on a Zumbo-Thomas
  weighted-least-squares (Pratt-Pregibon) partition (p. 333) — not the
  Nagelkerke pseudo-R² this package computes — and states them on the
  ONE-degree-of-freedom UNIFORM increment, not the 2-df omnibus this package
  previously lettered (`delta_r2`). Both mismatches were confirmed against
  the primary source (read in full via institutional access) and would need
  correcting together, but the replacement statistic is itself
  underdetermined by that source: eq. 4 (p. 333) does not say whether its
  correlation is taken against the observed response or the working response
  of the IRLS linearization, nor on which scale the standardized coefficient
  is computed, and both choices change the number. A package cannot letter a
  quantity it cannot compute, so the honest fix is "not applicable," not a
  guessed replacement. The bands are additionally scoped to the paper's own
  simulation — dichotomous responses generated from a 3PL model (pp. 337,
  339) on 40-item tests (pp. 336-337) — so they were never licensed for the
  polytomous logistic-regression sweep either (see PR #1890's independent
  finding on this point).
- **Migration.** Any stored or logged `jg_class` values from before this
  change reflected bands applied to the wrong quantity (2-df omnibus instead
  of 1-df uniform) computed from the wrong statistic (Nagelkerke instead of
  the Zumbo-Thomas WLS partition); they should not be treated as ground truth
  and do not need to be "corrected" to a new letter, because no letter is
  defensible. Use `delta_r2` / `delta_r2_uniform` (continuous, unchanged) and
  `flagged_bh` (significance only, unchanged) directly.
- **References.** Jodoin, M. G., & Gierl, M. J. (2001). Evaluating Type I
  error and power rates using an effect size measure with the logistic
  regression procedure for DIF detection. *Applied Measurement in Education,
  14*(4), 329-349. https://doi.org/10.1207/S15324818AME1404_2 — Nagelkerke,
  N. J. D. (1991). A note on a general definition of the coefficient of
  determination. *Biometrika, 78*(3), 691-692.
  https://doi.org/10.1093/biomet/78.3.691 — Zumbo, B. D. (1999). *A handbook
  on the theory and methods of differential item functioning (DIF)*.
  Directorate of Human Resources Research and Evaluation, Department of
  National Defense.

#### Require q_general/q_specific/q_nuisance, no unsourced defaults (#1933, refs AGENTS.md, #1929)

- **Breaking.** Per project rule (tuning numbers such as quadrature node counts must be caller arguments with no study-specific or unsourced default), four new APIs added this release cycle (#1873, #1888, #1889, #1912, #1925, #1926, #1928) had unsourced numeric defaults; those defaults are removed and the arguments are now required:
  - `fast_mlsirm.focal_expected_total_score_monotonicity`: `q_nuisance` (was `41`).
  - `fast_mlsirm.bifactor_expected_total_score_monotonicity`: `q_specific` (was `41`).
  - `fast_mlsirm.bifactor_grm.fit_bifactor_grm`: `q_general`, `q_specific` (were `21`, `11`).
  - `fast_mlsirm.bifactor_multigroup.fit_bifactor_grm_multigroup`: `q_general`, `q_specific` (were `21`, `11`), now required and keyword-only after `anchor_mask=None`.
  - `mlsirm_core::bifactor_grm::BifactorGrmConfig` no longer implements `Default`; every field must be set explicitly at every call site.
- Existing call sites (Python wrapper tests, two Rust loglik-only helpers, and the PyO3 binding) are updated to pass every field explicitly. Each touched call site gains a test asserting that omitting the node-count argument raises `TypeError` (Python) or fails to compile (Rust, no `Default`).

#### Customer copy actionability

- Public-API error paths now state the invalid input and the concrete next
  action instead of internal validation vocabulary: judge-scoring projection
  errors say to rebuild criterion mappings as plain dicts keyed by criterion id
  (replacing "exact built-in dict" jargon), Rust backend unavailability errors
  name the install/reference-path next steps, and unknown serving item codes
  point at the bundle's items list.
- CLI workspace-boundary and candidate-input errors append actionable next
  steps (move the file under the working directory, use unique
  `label=path.npy` candidate flags, reduce oversized candidate sets) without
  changing exit codes, validation order, or fail-closed behavior.
- Diagnostics report renderer errors tell the customer how to recover:
  choose a `.html` output name and regenerate unsupported JSON via
  `fast-mlsirm diagnose-fit` / `fast-mlsirm diagnose-dimensions`.
- README quickstart guidance no longer instructs an unexecutable command
  (`fit --backend numpy`; production backend choices are `{rust, auto}`) and
  now points to the explicit NumPy reference path (`--reference` /
  `fast_mlsirm.fit_reference`); feature copy describes fixed-item calibration
  by its method instead of legacy package names.

#### Lineage channel weight evidence

- Restore the `mlsirm-core` boundary to a domain-neutral criterion-anchor
  contract. The canonical field is `criterion_anchor`; the historical serialized
  `tepp_anchor` field remains readable only as a compatibility alias. Producer-
  specific schema validation is no longer compiled into the numerical core.

#### Release cut 0.9.1

- Project version is bumped to 0.9.1 in `pyproject.toml`, `crates/mlsirm-core`,
  and `crates/fast-mlsirm-py`. The accumulated `Unreleased` notes now form the
  `[0.9.1] - 2026-08-25` release section: new governed contracts (a judge
  construct-measurement contract for LLM-as-a-Judge orchestration, a Rust-owned
  independent longitudinal state layer, and a joint MAP hierarchical
  continuous-time AR(1) Rasch estimator), extended-precision identity
  preservation at the Rust boundary (Rasch CML control/group identity,
  G-theory mastery-cut identity, RSM tolerance identity through `f64`), a
  continuation of the hostile-callback/conversion-protocol hardening sweep
  across dozens of public entry points (CAT, ATA, CDM, DIF, equating, facets,
  fitting diagnostics, G-theory, inference, interaction maps, judge panels,
  linking, MHRM, Mokken, Oakes uncertainty, parallel analysis, polytomous
  prediction/recovery, Rasch CML, RSM, subscores, Warm WLE, and
  Benjamini-Hochberg admission, among others), governance/provenance fail-closed
  controls for release, buyer-evidence, PR queue, and procurement source-commit
  provenance, method-literature citation ADRs, and reproducibility work
  binding `uv.lock` resolution to the declared Python floor.
- This cut also removes seven authoritative fragments that no longer carried
  genuinely unreleased content: six whose content was already recorded verbatim
  in the `[0.9.0] - 2026-08-24` section by that release's fold but whose files
  were never deleted (`1028-fitstats-sx2-control-callback-safety.md`,
  `1032-gtheory-control-preflight.md`, `1266-gtheory-resource-admission.md`,
  `1268-gtheory-dstudy-row-bound.md`, `1269-gtheory-numpy-dstudy-controls.md`,
  `1314-linking-evidence-admission.md`), plus the standing predecessor note
  `release-0.9.0-cut.md`, whose substance is permanently recorded in that same
  section and in git history, mirroring the precedent set by the v0.9.0 cut's
  removal of the stale 0.8.0 leftover.
- Released authoritative fragments are removed from `docs/changelog.d`; the
  directory again holds only genuinely unreleased notes.

### Fixed

#### Compensatory 2PL response and tolerance admission

- Compensatory 2PL admission now seals both response evidence and the Rust `f64` convergence tolerance before caller data or native work. Response admission rejects caller-controlled array/numeric conversion protocols, complex evidence, ragged/non-numeric matrices, oversized logical matrices, and wider-precision values that would only become valid dichotomous observations after binary64 narrowing; exact NumPy numeric matrices and ordinary built-in matrices of concrete Python/NumPy 0/1/NaN values remain supported. The shared IRT minimum of two item columns is replayed from inert shape/row metadata after the existing logical-cell ceiling but before value-wise scans, scalar normalization, NumPy dense materialization, or compiled-core discovery, so structurally impossible one-item experiments cannot spend the full admitted response-work budget first. `tol` remains callback-safe but now also must preserve its exact numeric identity through normalization to Rust `f64`, so lossy extended-precision and large-integer controls fail closed while exact binary64-compatible controls remain supported. Accepted response evidence is normalized to package-owned contiguous `float64` before the existing Rust estimator boundary. The 2PL likelihood, quadrature, latent-correlation ECM, convergence arithmetic, and scoring are unchanged and remain Rust-owned.

#### Independent parameter provenance edges

- Item-parameter provenance now rejects self-referential edges: a provisional parameter artifact cannot name itself as its cold-start basis, and a calibrated parameter artifact cannot name itself as calibration evidence. This preserves an independent upstream provenance edge without changing any Rust-owned estimation, calibration, scoring, linking, uncertainty, or recovery arithmetic.

#### Item-bank lifecycle public identity replay

- Governed item-bank lifecycle records now replay their factory-sealed creation-time identity before returning a public fingerprint/id or serializing JSON-compatible content. A package-owned weak creation-seal registry binds each live factory-created object identity to its original fingerprint, so coherently rebinding both lifecycle content and the record's stored digest cannot manufacture fresh authority; dead-record entries are removed without retaining the record, and object-identity reuse is rejected. Callback-bearing mutations continue to fail closed, while valid lifecycle identities and payloads remain unchanged. No calibration, fit, DIF, information, linking, scoring, uncertainty, or other psychometric arithmetic changes.

#### Residual interaction-map structural budget

- Bound exact built-in matrix traversal independently of logical numeric cells so malformed empty-row fan-out fails before dense NumPy or compiled-core work while valid matrices inside the 20,000,000-cell evidence contract remain supported.

#### Finite-population proportion sampling design

- Replay the Rust `population_size <= 2^53` and 100,000-strata resource domains at the Python boundary before member normalization or Rust dispatch, and reject integer-valued strict probability controls without an unnecessary `float(...)` conversion so oversized/invalid controls fail with package-owned `ValueError` rather than consuming avoidable work or surfacing conversion overflow.

#### RSM response structural and shape budgets

- Bound exact built-in Rating Scale Model response-carrier traversal independently
  of logical numeric cells, so malformed empty-row fan-out cannot consume
  unbounded Python preflight work while keeping the response cell count at zero.
- Preserve the existing 20,000,000-cell RSM evidence envelope and every valid
  non-empty persons-by-items matrix inside it: built-in row plus scalar traversal
  is bounded by twice the logical-cell ceiling before NumPy materialization.
- Replay the established two-dimensional rectangular response contract and the
  minimum-two-item RSM/IRT design contract from inert ndarray shape or exact
  built-in row metadata after resource accounting but before value-wise scans or
  dense float64 marshalling. Small-backed 1-D and one-item broadcast views now
  fail their existing structural diagnostics without first allocating a large
  dense matrix.
- Keep RSM likelihood, marginal-ML EM/ECM, shared-threshold estimation, latent
  integration, scoring, convergence, and uncertainty arithmetic unchanged in the
  Rust core.

#### Sampling result contract replay

- Replay the Rust-owned sampling algorithm identity before public result marshalling so a same-schema stale or foreign extension fails closed instead of being exposed under the current contract.
- Validate every returned stratum inclusion-probability ratio against the Rust-returned sample and population counts before constructing the public sampling artifact, with stable package-owned errors for missing, malformed, count-mismatched, or inconsistent ratio evidence.
- Preserve the Rust-owned sample-size, finite-population-correction, proportional-allocation, and equal-cost Neyman arithmetic unchanged.

#### Bounded capped-strata allocation

- Replace repeated full active-set rescans in finite-population capped stratum allocation with a threshold-sorted water-filling pass, so the cap phase inspects each admitted stratum at most once after sorting.
- Add a maximum-envelope 100,000-strata census regression and an operation-count proof for the cap phase without relying on wall-clock timing.
- Preserve the existing Rust-owned proportional/Neyman quotas, census caps, deterministic input-order tie behavior, largest-remainder integerization, exact inclusion-probability ratios, and fail-closed zero-allocation contract.

#### Release acceptance watchdog budget

- Derive the commercial-release wrapper deadline for `release_acceptance.py` from the authoritative sequential inner acceptance budgets plus a 60-second orchestration margin, so future bounded-stage changes cannot silently reintroduce an outer watchdog that terminates legitimate fail-closed acceptance before the inner operation-specific deadline can report its evidence.

#### Release helper import integrity

- Fail closed when repository-owned bounded JSON or subprocess helpers raise an internal missing-dependency error; direct-script fallback now occurs only when the `scripts` package or the bounded helper module itself is unavailable, preserving the first causal boundary.

#### Confirmatory loading-pattern evidence admission

- Seal confirmatory items-by-dimensions loading-pattern evidence before NumPy array or numeric conversion protocols can execute, while preserving exact NumPy and ordinary built-in Boolean/integer/real 0/1 matrices as canonical read-only `int64` model structure.
- Reject callback-bearing providers/subclasses, non-real or non-numeric storage, ragged/non-2-D evidence, non-finite values, and non-binary values with package-owned diagnostics before model-resolution work.
- Replay the canonical read-only `int64` loading structure before public dimension or item-count resolution so post-construction field rebinding cannot execute caller-controlled shape metadata.

#### DIMTEST release-build diagnostic ownership

- Keep the per-group DIMTEST formula intermediates used by the independent Nandakumar & Stout oracle in test builds only, while release builds retain only the group contribution consumed by the production statistic. This removes the production dead-field warning without suppressing lints or changing DIMTEST arithmetic, public results, or the existing intermediate-value oracle.

#### Bound the 2PL slope's magnitude without constraining its sign (#1884, refs #1881 acceptance item 4)

- **Breaking, for any caller reading a `fit_mmle_2pl` slope that was previously clamped to the lower bound.** `mmle.rs` and `python/fast_mlsirm/estimators/mmle.py` clamped the Newton M-step slope update to `[1e-3, 10.0]`, which is also a hard positivity floor: a true negative slope (e.g. a reverse-keyed item, true value `-0.90`) was silently returned as `0.0010`, indistinguishable from an item that measures nothing. The bound is now symmetric (`[-10.0, 10.0]`), still guarding magnitude but no longer constraining sign.
- Since `(a, theta) -> (-a, -theta)` leaves the likelihood invariant, `fit_mmle_2pl` now canonicalizes on the largest-magnitude slope being positive (`canonicalize_reflection`), the same convention already used by `fit_grm`, `fit_gpcm`, `fit_twopl`, and `fit_mhrm`.

#### Bound the testlet/mixture slope's magnitude without constraining its sign (#1885, refs #1881)

- **Breaking, for any caller reading a `fit_testlet` or `fit_mixture` slope that was previously clamped to the lower bound.** Following the same audit that produced #1884, `testlet.rs` (Newton M-step and projection step) and `mixture.rs` (Newton M-step) each clamped their slope update to `[1e-3, 10.0]`, the same positivity floor that silently returned reverse-keyed slopes as `0.0010`. Both are now bounded symmetrically (`[-10.0, 10.0]`), guarding magnitude without constraining sign, and both canonicalize on the largest-magnitude slope being positive via the shared `canonicalize_reflection` introduced by #1884.
- `fit_testlet`'s testlet effect `gamma ~ N(0, sigma2)` and `fit_mixture`'s analogous structure are accounted for in the per-model reflection algebra, since the intercept/testlet terms are not all invariant under `(a, theta) -> (-a, -theta)` the same way across models.

#### Per-focal-group anchor sets and their intersection (#1891, addresses requirement 3 of #1874)

- Fix a defect in `dif_polytomous_purified`'s (#1890) purification loop surfaced while testing the `insufficient_anchor_items` failure mode: the loop's prior behavior on that termination path is corrected so it no longer returns a result inconsistent with a failed purification.

#### Fail closed when dedicated Statistical Studies filters match nothing (#1937, closes #1869)

- The dedicated Statistical Studies jobs ran `cargo test ... --exact <name>`, which exits 0 when the filter matches no test, so a renamed or removed recovery study would keep publishing a green true-parameter-recovery signal with zero tests actually run. Each of the four dedicated steps now captures its output and asserts exactly one test passed, so a filter that matches nothing now fails the job instead of silently reporting success.

#### Update logistic_dif_purified doc/test for retired jg_class (#1940)

- `logistic_dif_purified`'s docstring still claimed a non-uniform item is removed from its purification criterion, which #1880/#1935's retirement of `jg_class` to unconditional `"U"` made permanently impossible (the criterion can never fire, so the anchor never shrinks). The docstring is corrected to state the current no-op purification behavior and point callers at `mantel_haenszel_dif_purified` or the raw `delta_r2`/`flagged_bh` outputs for a real purified or significance-only read; the corresponding test assertion is updated to match.

## [0.9.1] - 2026-08-25

### Added

#### Judge construct measurement contract

- Add a package-level LLM-as-a-Judge measurement contract that treats each rubric criterion as one dichotomous or polytomous item, enforces zero-based category coding, and separates the three-item identification floor from the package's five-item default and seven-item recommended facet policy.
- Keep the package-wide facet ceiling at 11 items even when callers provide a custom `JudgeConstructPolicy`; custom policies may tighten the quality envelope but cannot silently raise the documented hard maximum.
- Persist the originating validated construct policy on package-created `JudgeConstructSpec` records and replay that policy at projection time, so post-construction criterion mutation cannot silently turn a below-minimum or above-maximum facet into an apparently policy-compliant handoff.
- Project judged responses into deterministic persons × items matrices only when every result carries exactly the non-blank criterion identities declared by the construct specification, and preserve `spec.criterion_ids` as the authoritative output-column order so downstream item parameters cannot be mislabeled by lexical key ordering.
- Pass the authoritative `spec.criterion_ids` order directly to a package-owned projection helper instead of relying on matching lexical sorts or mutating the shared `LLMJudgeResult.to_irt_row()` method at import time.
- Validate `item_type`, category-count semantics, and the `allow_short_form` Boolean control before iterating caller criterion evidence; preserve exact built-in and concrete NumPy Boolean short-form controls while rejecting callback-bearing truth-value providers without executing caller code.
- Revalidate direct and post-construction-mutated `JudgeConstructSpec` records at the projection trust boundary, including non-blank criterion identities, the global 3..11 item envelope, originating policy bounds when present, and dichotomous/polytomous category semantics, before reading item identities or marshalling rows.
- Keep GRM/GPCM fitting, likelihood, scoring, recovery, and all production psychometric arithmetic Rust-owned; the new surface performs validation, marshalling, and recovery evidence only.

#### Rust longitudinal state layer

- Added a Rust-owned independent per-respondent OLS trend and discrete-sequence
  AR(1) state predictor behind the sealed `fast_mlsirm.multilevel` contract.
- Preserved exact sequence gaps, missing-occasion output state, deterministic
  respondent sharding, RMSE/count diagnostics, and PyO3/Python marshalling.
- Documented the compatibility wire label `random_intercept_slope` as independent
  OLS with no population random-effects distribution or shrinkage, and the AR
  path as caller-supplied `phi` without coefficient estimation.
- Added slope-recovery, missingness, irregular-calendar/non-contiguous-sequence,
  worker-determinism, and fail-closed contract tests with APA 7 doctoring.
- This fragment does not claim full multilevel IRT random-effect integration,
  uncertainty, continuous-time transitions, or GPU recurrent-state parity.

#### Joint MAP hierarchical continuous-time AR(1) Rasch

- Added a Rust-owned joint MAP hierarchical continuous-time AR(1) Rasch
  estimator behind `fit_hierarchical_longitudinal_irt`, stacked on the
  `#976` longitudinal design handoff.
- Estimated shared population hyperparameters `(mu, tau, lambda)` and person-
  occasion states from exact millisecond elapsed-day gaps. State intervals
  are Wald intervals from measurement observed information; short series
  leave `lambda` weakly identified under joint MAP.
- Documented the estimand as joint MAP, not independent OLS, not caller-
  supplied discrete AR, not Fox and Glas Gibbs, and not estimated
  multiple-membership `u_h`. GPU parity is reported false because the
  existing wgpu path owns a different MLSIRM objective.
- Added multi-seed true-parameter recovery, irregular-time, missing-response,
  worker-determinism, and fail-closed marshalling tests with APA 7 ADR and
  doctoring.
- Normalized oversized integer and non-finite real execution controls before
  native dispatch so Python callers receive package-owned validation errors.
- Enforced finite, identified sum-zero Rasch item intercepts in the simulator
  before generating recovery data.
- Labeled hyperparameter intervals as conditional on fixed item/state nuisance
  blocks and aligned CT-AR gradients with the active variance branch.

### Changed

#### Validate Rasch CML controls before data materialization

- Validate `max_iter` and `tol` before caller-owned response or group arrays are materialized by the public Rasch CML and Andersen LR entry points.
- Reject complex-valued response matrices and Andersen group labels before `float64` coercion can discard imaginary components and silently admit altered data.
- Establish exact package-trusted response/group container and scalar identities before NumPy materialization so arbitrary `__array__` providers, ndarray/container/numeric subclasses, and object/text storage cannot execute caller conversion protocols while defining the scientific evidence analyzed by Rust.
- Preserve Andersen external group identities as exact package-owned integers before deterministic dense-ID construction, so distinct labels above the `float64` exact-integer boundary do not collapse and large unsigned labels do not wrap through signed narrowing.
- Reject response evidence above 20,000,000 logical cells before NumPy stacking, `float64` materialization, or signed-`int64` allocation, including oversized exact broadcast matrices and exact NumPy row leaves nested inside trusted built-in response matrices.
- Preserve exact NumPy Boolean/integer/unsigned/real arrays, ordinary built-in response rows, finite non-negative integral group labels, and supported concrete NumPy scalar compatibility inside the explicit response resource envelope.
- Keep conditional likelihood, optimization, information, LR, p-value, and all other production psychometric arithmetic in the Rust core.

#### Strengthen polytomous recovery calibration evidence

- Extend deterministic GRM, GPCM, CAT, and fixed-item-parameter recovery studies to require signed bias, MAE, finite positive Rust-returned posterior uncertainty, and empirical coverage of the normal-approximation interval `theta_eap ± 1.96 * theta_sd` alongside RMSE, while retaining correlation only as supplementary recovery evidence.
- Preserve CAT's independent adaptive-efficiency gate on mean administered items, so uncertainty calibration and error recovery cannot mask a fallback to non-adaptive item selection.
- Keep all production likelihood, marginal-ML/EM, EAP/CAT scoring, item-information/selection, stopping, and uncertainty arithmetic Rust-owned; the added Python calculations are explicit true-parameter recovery-test summaries only.

#### Harden observed-score logistic DIF controls

- Validate logistic and purified observed-score DIF semantic controls before caller-owned response/group materialization and before compiled Rust-core discovery.
- Reject caller-defined scalar subclasses, arbitrary conversion providers, booleans-as-numbers, invalid FDR levels, zero iteration caps, negative anchor floors, and values outside native `usize` without invoking caller callbacks.
- Preserve genuine supported NumPy scalar compatibility and keep all logistic/Mantel-Haenszel/purification statistics and BH arithmetic Rust-owned.

#### Method literature and citation ADRs

- Record primary-paper citations and ADRs for the shipped Angoff delta-plot
  DIF screen and Bradley–Terry / Hunter MM ranking estimators. These are
  psychometric method records, not new product capabilities and not
  CWE/OWASP/NIST controls.

#### Own residual interaction-map computation in the psychometric core

- Add a Rust-backed, complete-case Gabriel residual interaction-map contract
  for consumers that already hold observed responses and fitted IRT
  expectations. The API returns coordinates, singular values, axis inertia,
  reconstruction, unexplained residual, and the exact cross term without
  product identifiers, persistence, authorization, or presentation policy.

### Fixed

#### Harden serving-bundle callback boundaries

- Reject caller-defined serving-bundle container, schema-scalar, key, item-code, dimension-name, quadrature, and EAPsum-table subclasses before package validation can execute caller hashing, equality, iteration, comparison, or lookup callbacks. Valid bounded-JSON and exact built-in in-memory bundles keep the existing resource limits and Rust-owned scoring semantics.
- Reject serving-export factor identities that would require lossy complex/fractional/object coercion, signed narrowing, negative indices, or dimensions outside the supported `0..63` range before compiled-core discovery. Exact NumPy arrays and built-in sequences containing trusted integer-valued Python/NumPy scalars remain supported and are marshalled as contiguous `int64` identities.
- Normalize serving-export item identities from exact built-in list/tuple containers of exact built-in strings before artifact construction, preserving ordinary sequence compatibility while preventing caller container/string subclasses from executing callbacks or entering the frozen bundle.
- Normalize optional serving dimension labels from exact built-in lists of exact built-in strings and require their cardinality to match the admitted fitted dimension count before compiled-core discovery. Direct in-memory bundles replay the same label identity/cardinality contract after the historical bundle validator and before public scoring can discover Rust.
- Normalize trusted concrete NumPy integer `q_theta`/`q_xi` controls to built-in integers, require the serving-supported `{7,11,15,21,31,41}` quadrature domain, and replay the `q_xi ** latent_dim` and scoring-table allocation ceilings before compiled-core discovery so exported bundles cannot be born self-invalid or trigger oversized Rust table generation.
- Preflight serving-export latent-position (`zeta`) shape and numeric storage without arbitrary NumPy/container callbacks before deriving `latent_dim`, preserving exact numeric NumPy arrays and ordinary rectangular built-in list/tuple matrices while ensuring both forms hit the same serving-grid resource ceilings.
- Normalize trusted built-in/concrete NumPy real `eps_distance` controls to a built-in finite float and replay the existing positive serving-safe range before export, so returned and path-backed bundles self-validate/serialize while Boolean, non-finite, non-positive, and callback-bearing values fail before compiled-core discovery.
- Require the public `FitResult.convergence_status` evidence consumed by serving export to be an exact built-in string before the historical exporter can call `str(...)`, preventing caller-defined string-subclass conversion callbacks from bypassing the hardened artifact boundary.
- Make serving safety installation idempotent and partial-install recoverable so package reload/reinstallation cannot stack duplicate validation/export wrappers or drift error precedence over time.

#### RSM control and response admission safety

- Harden Rating Scale Model semantic-control admission so `n_cat`, quadrature size, iteration limits, and tolerance are normalized from trusted scalar identities before caller data or Rust capability work; hostile scalar subclasses and conversion/hash providers are rejected without callback dispatch while established built-in and NumPy compatibility remains unchanged.
- Reject arbitrary caller response array providers and container/numeric subclasses before NumPy protocol execution, while preserving exact NumPy arrays, ordinary built-in rows, and exact NumPy row arrays containing supported real numeric evidence, including exact NumPy Boolean scalar cells inside trusted built-in rows.
- Reject complex, object, and textual Rating Scale Model response storage before real-valued narrowing or caller element conversion, then marshal only admitted Boolean/integer/real numeric evidence to contiguous `float64` while preserving `NaN` missingness and the Rust-owned Andrich calibration semantics.

#### Fail closed on malformed PR queue evidence

PR queue governance scripts now require the repository-owned bounded JSON helper instead of falling back to a weaker inline decoder, and snapshot capture records malformed-payload errors when otherwise successful open-PR identity or history arrays contain non-object entries while preserving valid records and the raw identity count.

#### Require reconstructable procurement source commit provenance

- Make `scripts/build_procurement_due_diligence.py::_source_commit()` fail closed on every unreconstructable Git outcome: non-timeout command failures and executable/OS/subprocess errors now raise a stable package-owned `RuntimeError` instead of degrading to a `source_commit: "unknown"` placeholder, while keeping the existing bounded `GIT_METADATA_TIMEOUT_SECONDS` deadline for hung `git rev-parse HEAD` children.
- Accept only canonical full lowercase SHA-1 (exactly 40 hexadecimal characters) or SHA-256 (exactly 64 hexadecimal characters) object identities as the procurement source commit; empty, abbreviated, uppercase, non-hexadecimal, undersized, and oversized stdout are rejected with a stable package-owned error before any procurement due-diligence evidence can be emitted, so every published manifest cites a source that reconstructs the exact evidence build.
- Research basis: Ohm, Plate, Sykosch, and Meier (2020), *Backstabber's Knife Collection: A Review of Open Source Software Supply Chain Attacks*, DOI `10.1007/978-3-030-52683-2_2`. The methodological implication for this procurement path matches the release-evidence and PR queue governance paths: mutable or unverifiable provenance identities must never be silently substituted into consumed evidence, so identity resolution fails closed at the boundary where the value is produced.

#### Require reconstructable PR queue source commit provenance

- Make `scripts/build_pr_queue_governance.py::_source_commit()` fail closed on every unreconstructable Git outcome: non-timeout command failures and executable/OS/subprocess errors now raise a stable package-owned `RuntimeError` instead of degrading to a `source_commit: "unknown"` placeholder, while keeping the existing bounded `GIT_METADATA_TIMEOUT_SECONDS` deadline for hung `git rev-parse HEAD` children.
- Accept only canonical full lowercase SHA-1 (exactly 40 hexadecimal characters) or SHA-256 (exactly 64 hexadecimal characters) object identities as the governance source commit; empty, abbreviated, uppercase, non-hexadecimal, undersized, and oversized stdout are rejected with a stable package-owned error before any PR queue governance evidence can be emitted, so every published manifest cites a source that reconstructs the exact evidence build.
- Research basis: Ohm, Plate, Sykosch, and Meier (2020), *Backstabber's Knife Collection: A Review of Open Source Software Supply Chain Attacks*, https://doi.org/10.1007/978-3-030-52683-2_2. The methodological implication for this governance path matches the release-evidence path: mutable or unverifiable provenance identities must never be silently substituted into consumed evidence, so identity resolution fails closed at the boundary where the value is produced.

#### MH-RM response and control admission

- Reject complex-valued MH-RM response matrices before real-valued narrowing can discard imaginary response evidence.
- Establish a callback-free response-evidence boundary before NumPy materialization: exact NumPy arrays and ordinary built-in list/tuple trees containing package-trusted concrete Python/NumPy numeric scalars remain supported, while arbitrary array providers and caller-defined container/numeric subclasses fail closed before their protocols can execute. Exact numeric NumPy arrays nested as inert rows inside built-in containers remain compatible without admitting ndarray subclasses or object/text leaves.
- Replay the Rust-owned 200,000,000 persons×items response-cell ceiling before NumPy stacking, dense real-value narrowing, mask creation, or signed-integer marshalling; exact broadcast arrays and exact NumPy rows nested in trusted built-in matrices are charged by logical size, including repeated shared rows.
- Bound built-in response-tree structural traversal to twice the response-cell ceiling. Every valid non-empty rectangular persons×items sequence remains inside that work budget, while malformed empty or over-nested container fan-out can no longer consume unbounded Python traversal before NumPy materialization.
- Admit MH-RM family, iteration, proposal/tolerance, seed, and uncertainty/correlation controls before caller-owned response work or compiled-core discovery; normalize supported concrete Python/NumPy scalars to built-in Rust-boundary primitives and reject callback-bearing identities without executing their conversion protocols.
- Reject built-in and NumPy Boolean identities for the continuous `proposal_sd`, `target_accept`, and `tol` controls before response materialization or native discovery, while preserving Boolean semantics for `estimate_se` and `estimate_corr`.
- Mirror the Rust-owned unsigned iteration domains before response materialization, including negative/zero cycle and Metropolis-step values plus the full 64-bit `usize` conversion ceiling; values at or above `2**64` now fail with package-owned validation before PyO3 conversion, while the valid unsigned range through `2**64 - 1` remains lossless.
- Preserve documented `NaN` missingness, binary/GPCM category validation, and the existing Rust-owned MH-RM stochastic estimation, latent-correlation, uncertainty, convergence, and recovery arithmetic.

#### Mokken input admission

- Validate Mokken AISP scalar controls and score storage before compiled-core discovery, reject complex/object response evidence before numeric narrowing, and reject unsigned or floating category values outside signed `int64` before Rust marshalling.
- Reject caller-defined response array providers, container subclasses, and numeric subclasses before NumPy protocol execution while preserving exact NumPy arrays, ordinary built-in rows, and exact NumPy row arrays composed of supported real numeric evidence.
- Preserve the historical scalar semantics of exact zero-dimensional numeric NumPy arrays for `lower_bound` and `alpha` while continuing to reject ndarray subclasses, object/complex storage, booleans, and arbitrary caller conversion protocols.
- Keep unsigned signed-`int64` overflow detection exact across the supported NumPy 1.x/2.x range by comparing against an unsigned NumPy boundary instead of relying on value-based Python-int promotion.
- Reject response evidence above 20,000,000 logical cells before NumPy matrix materialization or signed-`int64` allocation, including oversized exact broadcast arrays and exact NumPy row leaves nested inside trusted built-in response matrices.
- Loevinger scalability, Z-statistics, and AISP arithmetic remain unchanged and Rust-owned.

#### Many-facet rating evidence admission hardening

- Reject complex or non-real-numeric Many-Facet Rasch response storage before real-valued marshalling so observed rating evidence cannot be silently projected onto different categories.
- Reject arbitrary top-level NumPy array providers and callback-bearing container/scalar identities before package-triggered array materialization, while preserving exact NumPy numeric arrays and ordinary exact built-in list/tuple evidence with trusted Python/NumPy real scalars.
- Bound Many-Facet Rasch response evidence to 20,000,000 logical cells before sequence materialization or dense real-valued work. Exact broadcast arrays are rejected from shape/size metadata, and built-in rating trees count trusted scalar leaves with nesting-depth-bounded traversal state before NumPy stacking.
- Bound built-in rating-tree structural traversal to three times the logical-cell ceiling, which preserves every valid non-empty rectangular 3-D input inside the 20,000,000-cell contract while preventing malformed empty-container fan-out from causing unbounded Python work before NumPy materialization.
- Reject ragged, mixed-depth, or empty built-in rating trees during the same callback-free preflight so the exact persons x items x raters rectangular shape is established before NumPy materialization.
- Preserve `NaN` missingness and existing category/domain validation for accepted real numeric arrays.
- Keep likelihood, marginal-ML EM, item difficulty, rater-severity, threshold, EAP, connectedness, and convergence arithmetic Rust-owned and unchanged.

#### Cognitive-diagnosis response admission hardening

- Reject complex or non-real-numeric response storage before real-valued marshalling across DINA/DINO, G-DINA, PVAF Q-matrix validation, Wald item-model selection, higher-order DINA/G-DINA, and shared/per-step-Q sequential G-DINA entry points so observed evidence cannot be silently projected onto different data.
- Require accepted numeric response evidence to round-trip exactly through the `float64` Rust boundary, rejecting extended-precision or integer values whose identity would change during marshalling while preserving exact values and `NaN` missingness.
- Reject callback-bearing response providers and caller-defined numeric/container identities before NumPy materialization while preserving exact NumPy arrays, exact built-in list/tuple trees, repeated/shared acyclic rows, inert NumPy-array rows, and package-known NumPy scalar evidence.
- Reject callback-bearing Q-matrix providers, container subclasses, and caller-defined numeric identities before NumPy materialization while preserving exact NumPy arrays and package-trusted built-in/NumPy numeric sequence evidence; repair the Q-matrix guard after direct `fast_mlsirm.cdm` reloads before public calibration reaches design validation.
- Seal per-step-Q sequential G-DINA design admission before NumPy protocols: accept only exact supported integer `n_steps` containers/scalars, mirror the Rust `SEQ_MAX_CAT = 50` bound before summing or touching step-Q evidence, and route `step_q` directly through the canonical callback-safe Q-matrix validator, including direct-module-reload coverage.
- Keep the response-container guard canonical across direct `fast_mlsirm.cdm` module reloads by delegating the reload fallback to the package safety implementation, so subclass rejection and shared-row compatibility cannot silently regress when package initialization is not re-run.
- Normalize model and stopping controls before caller response materialization across the CDM calibration, validation, model-selection, high…87725 tokens truncated…the persistent `theta`
  chain, and the averaged trait together — and once more at the end (mutation-verified: disabling the
  flip makes the reflection-fires test fail on all three sign checks). Loadings are UNCONSTRAINED so
  reverse-keyed / negative cross-loadings are representable. **Standard errors.** The Louis (1982)
  identity `I_obs = E[-d^2 l_c] - Var[d l_c]` gives per-item observed-information SEs, accumulated by
  a parallel RM filter (`sum_p (w_p - r_p^2) X_p X_p'`) over the convergence stage; where a
  finite-sample Louis block is not positive-definite the block falls back to the complete-data
  (Fisher) information (a conservative SE). **Guards.** A deterministic finite-difference anchor pins
  the per-item score and information against numerical derivatives of the complete-data logistic
  log-likelihood on an ASYMMETRIC D=2 cross-loader with a negative loading (catching sign, layout, and
  dims-map bugs a centered value-recovery test would not); the D=1 fit agrees with the established
  deterministic unidimensional MMLE (`mmle::fit_mmle_2pl`) within Monte-Carlo tolerance; a **D=6**
  recovery (3 pure anchors per dimension + a negative cross-loader, GH/QMC infeasible) recovers the
  loadings and per-dimension traits; the reflection-fires test drives a weak reverse-keyed pure anchor
  against a strong positive cross-loader so raw MH-RM lands the anchor negative and canonicalization
  must fire; and `validate` rejects rotationally-degenerate patterns, non-binary responses, and
  `burn_in >= max_cycles`. This first release fits the ORTHOGONAL 2PL (`Sigma = I`); a free latent
  correlation matrix and the polytomous item families are natural extensions of the same loop. Compute
  lives in `mlsirm_core::mhrm::fit_mhrm`; exposed to Python as `fit_mhrm` / `MhrmFit` via the
  `model=` specification API.

- **Confirmatory MULTIDIMENSIONAL generalized partial credit model** (Muraki, 1992).
  `fit_gpcm(responses, n_cat, model=...)` fits ORDERED polytomous categories with a SINGLE
  multidimensional discrimination vector per item and INTEGER category scores, completing the
  polytomous-MIRT trio (`fit_nominal` / `fit_grm` / `fit_gpcm`). Item `i` has a free slope `a_i` (free
  on the confirmatory 0/1 loading pattern from `model=models.confirmatory(...)`, items x D) and
  `n_cat-1` category step intercepts `gamma_i`, with `psi_k = k * (sum_{d in S_i} a_id theta_d) +
  gamma_i,k`, `gamma_i,0 = 0` pinned, and `P(Y_i = k | theta) = softmax_k(psi_k)`, `theta ~ MVN(0,
  I_D)`. This is the `a_ikd = k a_id` INTEGER-scoring restriction of the multidimensional nominal
  model in a distinct single-slope parametrization — NOT a mode of `fit_nominal` (which optimizes free
  per-category slopes), so it warrants its own estimator; and it is the ADJACENT-category-logit
  counterpart of the cumulative `fit_grm`. Unlike the GRM's thresholds, the GPCM steps are UNORDERED
  (the softmax is finite for any real `gamma`, so no ordering constraint exists or is imposed). It
  reduces to the unidimensional GPCM (`poly::fit_poly_unidim(PolyModel::Gpcm)`) at `D = 1` (within
  optimizer tolerance and up to reflection — NOT bit-exact, because `fit_poly_unidim` forces `a > 0`
  via a `log a` parametrization while the confirmatory model uses an UNCONSTRAINED slope so
  reverse-keyed / negative cross-loadings are representable). Estimated by Bock-Aitkin marginal MLE
  over the D-dim latent grid, REUSING the compensatory-MIRT node machinery (`nodes::build_xi_nodes`):
  `node_rule = "gh"` uses the `q^D` Gauss-Hermite grid (`D <= 3`), `"qmc"`/`"mc"` use `xi_points`
  Halton / Monte-Carlo draws (`D <= 6`, Jank 2005 QMC-EM), and the GPCM softmax cell of
  `poly::gpcm_logprobs` / `gpcm_node_gradient`. The per-item M-step is a finite-difference-Hessian
  Newton over `[a_{d0}..a_{d,L-1}, gamma_1..gamma_{M-1}]`, byte-for-byte the ascent of
  `poly::m_step_item` (ridge = Hessian conditioning only, not a prior), with the GPCM node gradient
  chained to the multidimensional slope (`d/da_id = sum_node g_base theta_d`, `d/dgamma_j = sum_node
  g_intercepts[j]`). Category scores are FIXED integers `0..n_cat-1` (that fixity is what makes the
  model GPCM rather than nominal), so the free per-category slope gradient returned by the shared cell
  (`g_scores`) is DROPPED — only the single `base` slope and the step intercepts are estimated. Init is
  `gamma_k = ln(freq_k / freq_0)` (a plain marginal log-odds, NOT a cumulative GRM-style boundary). EM
  uses the SIGNED monotonic-decrease stopping guard (a likelihood decrease errors, not the
  compensatory MIRT's `.abs()` check). **Identification.** Unit trait variances + a PURE
  single-dimension anchor item per dimension pin the rotation to the coordinate axes; the per-dimension
  reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` leaves `base` — hence every step and category
  probability — INVARIANT, so it is CANONICALIZED (as for the GRM / compensatory MIRT, and unlike the
  nominal, whose per-category slopes make the anchor sign ambiguous): dimension `d` is flipped so its
  largest-magnitude pure anchor loads positively, negating that dimension's slope column AND the trait
  `theta_d` but NOT the steps. `validate` rejects a rotationally-degenerate pattern (no pure anchor),
  an out-of-range category, and ANY unobserved category for an item, with a `nodes x items x n_cat`
  count-table cap and the rule-dependent D / q / xi_points bounds. **Guards.** The D=1 anchor recovers
  `fit_poly_unidim(Gpcm)`'s slope and steps within tolerance; a deterministic finite-difference anchor
  pins every per-(dimension, step) gradient slot on a fixed node set at D=2 (GH) AND D=4 (Halton) with
  a NON-IDENTITY dims map, M>=4 categories, deliberately NON-MONOTONE step values (unordered steps have
  no ordering canary, so the anchor exercises the free-step estimator directly) and distinct random
  per-category counts; because that FD anchor is map-invariant, a SEPARATE deterministic
  objective-value assertion at D=4 (dims `[0,2,3]`) pins the node-column dims map by computing
  `base = sum_t a_t node[dim_t]` and the GPCM log-probabilities BY HAND with LITERAL integer scores and
  matching the estimator's internal value to `< 1e-9` (the QMC path is never exercised by the D<=3
  recovery / MC); a reflection-FIRES test is constructed so the RAW EM mode lands the pure anchor
  NEGATIVE (a WEAK reverse-keyed pure anchor plus a STRONG positively-keyed cross-loader that dominates
  the dim0 orientation), so canonicalization MUST fire — asserting the anchor ends positive, the
  co-loader ends negative, the trait axis is sign-flipped (theta correlates negatively with the truth
  on the reflected dimension), and the steps are unchanged; mutation-verified (disabling the flip fails
  all three sign checks). A D=2 recovery carries a genuinely NEGATIVE cross-loader on a
  positively-anchored dimension (asserted `< -margin`) and recovers the unordered steps by RMSE. A
  Monte-Carlo (`D in {2, 3}`, pure anchors + sign-varied cross-loaders, `n_cat = 4`, GH `q = 15/11`,
  `N = 2500/2000`) recovers the loadings near-unbiased under a normal trait (loading RMSE ~0.08-0.09,
  bias ~0.00-0.01; step RMSE ~0.06-0.07) with the expected mild attenuation under a
  per-dimension-standardized right-skew trait (loading RMSE ~0.10-0.11, bias ~-0.04; step RMSE ~0.14),
  per-dimension trait EAP correlation ~0.74-0.77 and 100% convergence, EM monotone every replication
  (40-replication pilot; the committed `#[ignore]` test runs 500). Compute lives in
  `mlsirm_core::gpcm::fit_gpcm`; exposed to Python as `fit_gpcm` / `GpcmFit`.

- **Confirmatory MULTIDIMENSIONAL graded response model** (Samejima, 1969; Muraki & Carlson, 1995).
  `fit_grm(responses, n_cat, model=...)` fits ORDERED polytomous categories with a SINGLE
  multidimensional discrimination vector per item and ordered category boundaries: item `i` has a
  free slope `a_i` (free on the confirmatory 0/1 `loading_pattern`, items x D) and `n_cat-1` ORDERED
  boundary intercepts `beta_i`, with `P(Y_i >= k | theta) = sigmoid(sum_{d in S_i} a_id theta_d +
  beta_i,{k-1})`, `theta ~ MVN(0, I_D)`. This is the ORDERED counterpart of the multidimensional
  nominal model and the polytomous generalization of the compensatory MIRT; it reduces to the
  unidimensional GRM (`poly::fit_poly_unidim(PolyModel::Grm)`) at `D = 1` (within optimizer tolerance
  and up to reflection — NOT bit-exact, because `fit_poly_unidim` forces `a > 0` via a `log a`
  parametrization while the confirmatory model uses an UNCONSTRAINED slope so reverse-keyed / negative
  cross-loadings are representable). Estimated by Bock-Aitkin marginal MLE over the D-dim latent grid,
  REUSING the compensatory-MIRT node machinery (`nodes::build_xi_nodes`): `node_rule = "gh"` uses the
  `q^D` Gauss-Hermite grid (`D <= 3`), `"qmc"`/`"mc"` use `xi_points` Halton / Monte-Carlo draws
  (`D <= 6`, Jank 2005 QMC-EM), and the GRM cumulative-logit cell of `poly::grm_logprobs` /
  `grm_node_gradient`. The per-item M-step is a finite-difference-Hessian Newton over
  `[a_{d0}..a_{d,L-1}, beta_1..beta_{M-1}]`, byte-for-byte the ascent of `poly::m_step_item` (ridge =
  Hessian conditioning only, not a prior), with the GRM node gradient chained to the multidimensional
  slope (`d/da_id = sum_node g_base theta_d`, `d/dbeta_j = sum_node g_thr[j]`). The ORDERED-threshold
  constraint is maintained WITHOUT an explicit reparametrization: every adjacent boundary pair is a
  middle category whose log-probability goes non-finite the instant the pair inverts (`0*NaN=NaN` so a
  zero expected count cannot mask it), so the backtracking line search — which rejects any non-finite
  step — keeps `beta` fully ordered by adjacency + transitivity. EM uses the SIGNED
  monotonic-decrease stopping guard (a likelihood decrease errors, not the compensatory MIRT's
  `.abs()` check). **Identification.** Unit trait variances + ordered thresholds + a PURE
  single-dimension anchor item per dimension pin the rotation to the coordinate axes; the
  per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` leaves `base` — hence every
  threshold and category probability — INVARIANT, so it is CANONICALIZED (unlike the nominal, whose
  per-category slopes make the anchor sign ambiguous): dimension `d` is flipped so its
  largest-magnitude pure anchor loads positively, negating that dimension's slopes AND the trait
  `theta_d` but NOT the thresholds. `validate` rejects a rotationally-degenerate pattern (no pure
  anchor), an out-of-range category, and ANY unobserved category for an item (a GRM boundary would
  diverge), with a `nodes x items x n_cat` count-table cap and the rule-dependent D / q / xi_points
  bounds. **Guards.** The D=1 anchor recovers `fit_poly_unidim(Grm)`'s slope and thresholds within
  tolerance (all-positive DGP, the domain where its `log a` is correctly specified); a deterministic
  finite-difference anchor pins every per-(dimension, threshold) gradient slot on a fixed node set at
  D=2 (GH) AND D=4 (Halton) with a NON-IDENTITY dims map, M>=4 categories, STRICTLY-DECREASING
  thresholds (gaps >> the FD step, since the GRM cell NaNs on an inverted boundary) and distinct
  random per-category counts; because that FD anchor is map-invariant, a SEPARATE deterministic
  objective-value assertion at D=4 (dims `[0,2,3]`) pins the node-column dims map by computing
  `base = sum_t a_t node[dim_t]` and the GRM log-probabilities BY HAND and matching the estimator's
  internal value to `< 1e-9` (the QMC path is never exercised by the D<=3 recovery / MC); a
  reflection-FIRES test drives a reverse-keyed largest pure anchor and asserts it ends positive, a
  co-loader ends negative, and the thresholds are unchanged and still ordered; a D=2 recovery carries
  a genuinely NEGATIVE cross-loader on a positively-anchored dimension (asserted `< -margin`) with
  strictly-ordered recovered thresholds. A Monte-Carlo (`D in {2, 3}`, pure anchors + sign-varied
  cross-loaders, `n_cat = 3`, GH `q = 15/11`, `N = 2500/2000`) recovers the loadings near-unbiased
  under a normal trait (loading RMSE ~0.10, bias ~0.00-0.01; threshold RMSE ~0.05-0.06) with the
  expected mild attenuation under a per-dimension-standardized right-skew trait (RMSE ~0.17/0.18,
  bias ~-0.12/-0.13), per-dimension trait EAP correlation ~0.63-0.70 and 100% convergence, EM
  monotone and thresholds ordered every replication (40-replication pilot; the committed `#[ignore]`
  test runs 500). Compute lives in `mlsirm_core::grm::fit_grm`; exposed to Python as
  `fit_grm` / `GrmFit`.
- **Confirmatory MULTIDIMENSIONAL nominal response model** (Bock, 1972; Thissen, Cai, & Bock,
  2010). `fit_nominal(responses, n_cat, model=...)` fits unordered polytomous categories
  with CATEGORY-SPECIFIC multidimensional discrimination: category `k` of item `i` has a free slope
  vector `a_ik` (free on the confirmatory 0/1 `loading_pattern`, items x D) and intercept `c_ik`,
  and `P(Y_i = k | theta) = softmax_k(sum_{d in S_i} a_ikd theta_d + c_ik)` with the baseline
  category `0` pinned `a_i0 = 0, c_i0 = 0`, `theta ~ MVN(0, I_D)`. This generalizes the
  unidimensional `poly::fit_nominal` to D latent dimensions, and reduces to it EXACTLY at `D = 1`
  (the same general free-`a_k` parametrization). Estimated by Bock-Aitkin marginal MLE (EM) over the
  D-dimensional latent grid, REUSING the compensatory-MIRT integration machinery: `node_rule = "gh"`
  uses the `q^D` Gauss-Hermite product grid (`D <= 3`); `"qmc"`/`"mc"` use `xi_points` Halton /
  Monte-Carlo draws (`D <= 6`), the quasi-Monte-Carlo EM of Jank (2005). The per-item M-step is a
  Newton on the concave multinomial-logit complete-data objective, byte-for-byte the
  finite-difference-Hessian ascent of `poly::nominal_m_step` (the ridge is Hessian conditioning only,
  NOT a parameter prior, so the fit is genuine MML and the D=1 reduction is bit-exact), generalized
  so the softmax residual `resid_k = r_k - n P_k` drives `d/dc_ik = sum_node resid_k` and
  `d/da_ikd = sum_node resid_k theta_d`. EM uses `fit_nominal`'s relative-tolerance stopping with a
  SIGNED monotonic-decrease guard (a likelihood decrease errors, rather than the compensatory MIRT's
  `.abs()` check which would accept one as convergence). **Identification.** Baseline category +
  unit trait variances + a PURE single-dimension anchor item per dimension pin the rotation to the
  coordinate axes: a pure anchor forces every one of its category slopes onto the axis, so an
  orthogonal trait rotation must send that axis to `+-e_d`, and the confirmatory labels forbid axis
  permutation — leaving only a per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)`,
  which (as in `fit_nominal`) is NOT canonicalized; recovery is assessed up to it. `validate` rejects
  a rotationally-degenerate pattern (no pure anchor), an out-of-range category, and — a guard
  `fit_nominal` lacks — ANY unobserved category for an item (its intercept would diverge and its D
  slopes be unidentified), plus a `nodes x items x n_cat` count-table cap and the rule-dependent
  D / q / xi_points bounds. **Guards.** The D=1 anchor reproduces `fit_nominal`'s scores/intercepts
  and whole loglik trace bit-exactly (< 1e-9); a deterministic finite-difference anchor pins EVERY
  per-(category, dimension) gradient component on a fixed node set at D=2 (GH) AND D=4 (Halton) with
  a NON-IDENTITY dims map and distinct random per-category counts (catching a category<->dimension
  transposition the D=1 reduction cannot see); a D=2 recovery carries a genuinely NEGATIVE
  cross-loader slope AND two OPPOSITE-sign sibling categories on the same dimension (catching a
  collapse of the free per-category slopes to a shared scalar discrimination); and baseline /
  off-pattern entries are asserted EXACTLY `0.0` with a free-parameter-count invariant. A
  Monte-Carlo (`D in {2, 3}`, pure anchors + sign-varied cross-loaders, `n_cat = 3`, GH
  `q = 15/11`, `N = 2500/2000`, assessed up to per-dimension reflection) recovers the category
  slopes near-unbiased under a normal trait (slope RMSE ~0.12 at `D = 2` / ~0.13 at `D = 3`, bias
  ~0.00-0.01) with the expected mild attenuation under a per-dimension-standardized right-skew trait
  (RMSE ~0.21/0.22, bias ~-0.09), per-dimension trait EAP correlation ~0.61-0.67 and 100%
  convergence, EM monotone every replication (the figures are a 40-replication pilot; the committed
  `#[ignore]` test runs 500). Compute lives in
  `mlsirm_core::nominal::fit_nominal`; exposed to Python as `fit_nominal` /
  `NominalResponseFit`.

- **Confirmatory compensatory multidimensional 2PL (MIRT), orthogonal or correlated**
  (Reckase, 2009; Bock, Gibbons, & Muraki, 1988).
  `fit_2pl(responses, model=...)` fits
  a general COMPENSATORY multidimensional 2PL in which an item may load FREELY on several
  latent dimensions, which trade off ADDITIVELY inside a single logit:
  `P(X_ij=1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`, `theta_j ~ MVN(0, I_D)`,
  where `S_i` is item `i`'s loading set from a 0/1 confirmatory pattern (items x dimensions).
  This is Reckase's compensatory M2PL / the full-information item factor model, distinct from
  the existing simple-structure `Mirt` (one dimension per item) and the orthogonal bifactor
  (one primary + one general per item): arbitrary within-item cross-loadings break the
  simple-structure quadrature factorization, so it is a dedicated estimator (standalone
  `mlsirm_core::twopl`) with the full `q^D` product Gauss-Hermite grid (`D <= 3`). Estimated by
  marginal-ML EM: the E-step is streamed per person (no `N x q^D` posterior materialized), and
  each item M-step is an `(n_i + 1)`-dimensional Newton generalizing `fit_mmle_2pl`'s 2x2 — the
  ridged, positive-definite `-Hessian` block solved by Gaussian elimination with a backtracking
  line search that keeps the marginal loglik monotone. Loadings are **not** constrained
  non-negative (reverse-keyed and suppressor cross-loadings are representable); the
  per-dimension sign is fixed by a reflection anchor. **Latent traits:** `theta ~ MVN(0,
  Sigma)` — orthogonal (`Sigma = I`) by default, or with `estimate_corr = true` the
  inter-factor **correlation matrix is estimated**: the standard grid is mapped through
  `chol(Sigma)` (`theta_g = L z_g`, a measure-preserving change of variables that reuses the
  product-GH weights and the item M-step verbatim), and the `D(D-1)/2` free correlations ascend
  the Gaussian-prior objective `-0.5[log|Sigma| + tr(Sigma^{-1} C)]` (`C` the posterior second
  moment, accumulated via the per-node marginal mass so it adds nothing to the E-step order)
  with backtracking + a full-matrix positive-definite guard, keeping EM monotone; the reflection
  anchor also negates the flipped dimension's correlation off-diagonals. A deterministic
  finite-difference anchor pins the correlation gradient (`D=2` and `D=3`); a known-`Sigma`
  (`rho=0.5`) recovery with a reflection-triggering negative anchor confirms the sign flip; and
  a 500-rep MC recovers the correlations essentially UNBIASED against the realized sample
  correlation (correlation RMSE ~0.035-0.05, bias ~0.0005 under the normal model / ~0.017 under
  the NORTA right-skew arm), 100% convergence with every fitted `Sigma` strictly interior.
  `D > 3` (coarser GH or QMC) remains deferred. Identification is enforced by
  `validate`: every dimension must have a PURE single-loading anchor item, so
  rotationally-degenerate patterns (e.g. all-ones) are rejected rather than returning a point
  on a non-identified ridge. Verified with the N(0,I) grid-moment identities, a DETERMINISTIC
  finite-difference anchor pinning the full item gradient AND the off-diagonal cross-Hessian
  (the local->pattern-dimension map) to `< 1e-4`, an exact reduction to `fit_mmle_2pl` at `D=1`
  (`gh_rule(41)` is the same grid; loadings/intercepts agree to `< 1e-2`), and a non-trivial
  `D=2` recovery with asymmetric loadings INCLUDING genuinely negative ones (recovered with
  correct sign). A 500-replication Monte-Carlo (`D in {2,3}`, `N = 3000/2000`, confirmatory
  pattern with pure anchors + cross-loaders) recovers the loadings essentially UNBIASED under
  the correctly-specified normal trait (loading RMSE ~0.10 at `D=2` / ~0.12 at `D=3`, bias
  ~0.006) and shows the expected mild loading attenuation under a per-dimension-standardized
  right-skew trait (shape misspecification; RMSE ~0.12/0.16, bias ~-0.06/-0.10), with
  per-dimension trait EAP correlation ~0.67-0.72 and 100% convergence, EM monotone every
  replication. Exposed to Python as `fit_2pl` / `TwoPlFit`.
- **`D > 3` confirmatory compensatory MIRT via quasi-Monte-Carlo EM** (Jank, 2005). The
  compensatory MIRT above was capped at `D <= 3` by its `q^D` Gauss-Hermite product grid;
  `fit_2pl` now takes a `node_rule` (`"gh"` default, or `"qmc"`/`"mc"`) that swaps
  the E-step integration nodes for a **Halton quasi-Monte-Carlo** (or seeded Monte-Carlo) rule,
  reaching `D = 4, 5, 6` (the Halton prime axes). This is Jank's (2005) QMC-EM: the E-step integral
  `int p(x|theta) phi(theta) dtheta` is evaluated at `xi_points` points drawn from the prior
  (Halton radical inverse mapped through the inverse-normal CDF, equal weights `1/xi_points`)
  instead of the product grid, and the node set is built ONCE before the EM loop, so the per-item
  `(n_i+1)`-dim Newton M-step and the correlated-`Sigma` ECM step are byte-for-byte the same code on
  the swapped nodes. The reused node generator (`mlsirm_core::nodes::build_xi_nodes`, shared with the
  marginal QMC-EM family) is parity-tested; its Gauss-Hermite arm is bit-identical to the existing
  product grid, so the `"gh"` path is unchanged bit-for-bit and every prior MIRT test passes verbatim.
  Both the orthogonal and the correlated-`Sigma` (Cholesky node-map `theta_g = L z_g`) paths carry
  over to `D > 3`. **Monotonicity.** With `Sigma = I` the nodes never move, so the orthogonal fit is
  monotone in the QMC-approximated marginal likelihood; the correlated `Sigma` M-step reparametrizes
  the node cloud, so that fit is monotone only up to the QMC quadrature error (overall ascent with
  per-step wobble ~1e-5 relative that shrinks as `xi_points` grows) — use the orthogonal path or a
  larger `xi_points` when strict monotonicity matters. Validation is rule-dependent: `"gh"` keeps
  `D <= 3` and the `q^D <= 200_000` node cap; `"qmc"`/`"mc"` cap `D <= 6` (the Monte-Carlo node
  builder has no internal cap, so this bound is its sole guard) and bound `xi_points`
  (`1..=200_000`, with checked `xi_points * n_items` and `xi_points * n_dims` allocations); `q`
  applies only to `"gh"`, `xi_points`/`xi_seed` only to `"qmc"`/`"mc"`. **Guards.** Beyond the
  reused-grid regression, a deterministic layout pin asserts `build_xi_nodes(Halton).grid[j*D+k] ==
  inv_normal_cdf(radical_inverse(j+1, prime_k))` at `D = 4` (independently fixing the prime-to-axis
  assignment, the index skip, and the row-major layout that a value-recovery test cannot see); the
  QMC weights are pinned to `-ln(n)` (invisible to every fit-level test since they cancel in the
  self-normalized posterior); a deterministic finite-difference anchor pins the analytic gradient and
  full cross-Hessian on a FIXED Halton grid at `D = 4` with a non-identity `dims` map; the reduction
  anchor is TWO-SIDED (a `D = 2` QMC fit agrees with the GH fit within QMC error AND differs
  bit-wise, so a silent GH fallback is caught); and the reflection anchor is exercised with a
  reverse-keyed largest anchor. **Accuracy.** A Monte-Carlo (`D in {4, 5}`, confirmatory pattern with
  pure anchors + alternating-sign cross-loaders, Halton `xi_points = 4000/6000`, `N = 2000/1500`)
  under a correctly-specified normal trait recovers the loadings near-unbiased (loading RMSE ~0.13 at
  `D = 4` / ~0.17 at `D = 5`, bias ~0.01) and shows the expected mild attenuation under a
  per-dimension-standardized right-skew trait (shape misspecification; RMSE ~0.16/0.21, bias
  ~-0.07/-0.09), with per-dimension trait EAP correlation ~0.58-0.64 and 100% convergence, EM
  monotone every replication (the reported figures are a 50-replication pilot; the committed
  `#[ignore]` test runs 500). QMC carries an `O(N^{-1} (log N)^D)` finite-node bias that grows with
  `D` (the higher-prime Halton axes degrade), so `D = 5, 6` and the correlated `Sigma` off-diagonals
  need materially larger `xi_points`; `xi_seed` (nonzero by default) applies a Cranley-Patterson
  shift that partly de-correlates the higher axes. Exposed to Python as `fit_2pl(...,
  node_rule=, xi_points=, xi_seed=)`.
- **Shared-Q sequential G-DINA for polytomous responses** (Ma & de la Torre, 2016;
  Tutz, 1990). `fit_seq_gdina(responses, q_matrix)` fits ordered polytomous cognitive
  diagnosis by the sequential (continuation-ratio) model: each ordered *step*
  `k in 1..=M_i` of item `i` has a continuation probability `s_ik(l) = P(X_i >= k | X_i
  >= k-1, reduced class l)` that is a saturated G-DINA over the item's `2^{K_i}` reduced
  attribute classes, and the category probabilities are the sequential decomposition
  `P(X_i = k | l) = (prod_{v<=k} s_iv(l))(1 - s_{i,k+1}(l))` with the stop sentinel
  `s_{i,M_i+1} = 0` (top category has no trailing factor — never eps-clamped, so its
  probability carries no spurious bias). Because the sequential likelihood factorizes
  into independent per-step Bernoullis on the at-risk set, the M-step is the closed-form
  saturated ratio `s_ik(l) = R_ik(l)/I_ik(l)` with `R` = expected count reaching category
  `>= k` and `I` = expected count reaching `>= k-1` — exactly `fit_gdina`'s saturated step
  on continuation counts. The population is a free profile distribution `pi_c`; `M_i` is
  derived as each item's maximum observed category (an item stuck at category 0 is
  rejected; a zero-frequency *interior* category is fine — it just means `s_{i,k+1} ~ 1`).
  With one step per item (binary data) it reduces to `fit_gdina` **bit-for-bit** (shared
  monotone init, identical E-step logprobs and closed-form ratio; a regression test
  asserts the whole loglik trace and step probs agree to `< 1e-12`). Deterministic anchors
  pin the sequential core with no Monte-Carlo noise: the category-probability identity
  (`P(0)=1-a, P(1)=a(1-b), P(2)=a*b`, non-centered) and the at-risk-count identity
  (responses `{0,1,1,2}` -> `s_1 = 3/4, s_2 = 1/3`, exercising the `{>=k}/{>=k-1}`
  denominator). A 500-replication Monte-Carlo (K=3, mix of M=2/M=3 items, N=2500, under
  BOTH a normal and a right-skew higher-order attribute distribution) recovers the model
  with category-probability RMSE ~0.020, at-risk-mass-weighted step RMSE ~0.020, and
  attribute-classification agreement ~0.97 — essentially identical across the normal and
  skew conditions, because the free `pi_c` nests the higher-order-implied distribution
  (no prior misspecification). **Scope:** this is the *shared item-level Q-vector*
  sequential G-DINA — every step of an item is a saturated G-DINA over the SAME required
  attributes; it is a restriction of Ma & de la Torre's general per-step (`q_ik`) model,
  whose step-distinct attribute requirements are a deferred non-goal (supply each item's
  Q-vector as the union of its steps' attributes). Compute lives in
  `mlsirm_core::cdm::fit_seq_gdina` (reuses `reduce_class`, the profile-grid posterior,
  and the saturated closed-form ratio); exposed to Python as `fit_seq_gdina` with the
  `SeqGdinaFit` wrapper (`item_step_prob` / `item_cat_prob` ragged accessors).
- **Per-step-Q sequential G-DINA — the full restricted-Q model** (Ma & de la Torre,
  2016). `fit_seq_gdina_qr(responses, step_q, n_steps)` lifts the restriction above: each
  ordered *step* `k` of item `i` carries its OWN attribute requirement `q_ik` (the paper's
  headline generality — step 1 may need attribute A, step 2 need A AND B), supplied as a
  row-major `(sum_i M_i) x K` restricted Q-matrix `Q_r`. The sequential factorization is
  unchanged, so each step is still an independent saturated Bernoulli on its at-risk set
  and the closed-form ratio `s = R/I` is still the exact complete-data MLE — but now each
  step's success is a saturated G-DINA over ITS OWN `2^{|q_ik|}` reduced classes. **Storage
  is union-class-indexed and lossless:** response probabilities depend only on the item's
  UNION `u_i = OR_k q_ik`, so the E-step posterior and the category probabilities are
  indexed by the `2^{|u_i|}` union reduced class (no `N x 2^K` materialization), while each
  step's own reduced class is computed DIRECTLY from the full profile `c` via
  `reduce_class(c, q_ik)` — never a union-mask AND, which would silently mis-gather the
  renumbered set bits. Step probabilities are stored step-row-major (`spo` over `sum_i M_i`
  rows, width `2^{|q_ik|}` each; `step_off[i]` per item, `step_kq[g] = |q_ik|`), category
  probabilities item-major over the union class. **Reduction guard:** giving every step of
  an item the item's Q reproduces `fit_seq_gdina` BIT-EXACTLY (layout-aware cell compare of
  the transposed step tables plus direct compare of the class-major category probs and the
  whole loglik trace — difference exactly `0`). A structural anchor (step 1 `q={A}`, step 2
  `q={A,B}`) asserts the per-step block widths are `2` and `4` (not one collapsed union
  block) and `n_parameters` reflects the per-step widths — a discrimination value recovery
  alone cannot make, since an over-collapse to the union would still fit — while recovering
  a large step-2 B-contrast (`s_2(A1,B0)=0.20` vs `s_2(A1,B1)=0.80`, gap >= 0.4) that the
  shared-Q model cannot represent. `validate` rejects an all-zero step row (a step measuring
  nothing), an attribute required by no step (all-zero union column), and `n_steps[i]` not
  equal to both the declared step count and the maximum observed category, with checked
  `(sum_i M_i) * K` and `2^{|u_i|}` allocations and the same `K` cap. A 500-replication
  Monte-Carlo (K=3, step-distinct M=2/M=3 items plus single-attribute M=1 identification
  items pinning each dimension, N, under BOTH a normal and a right-skew higher-order
  attribute distribution) recovers the model with at-risk-mass-weighted step-probability
  RMSE ~0.017, category-probability RMSE ~0.018, and attribute-classification agreement
  ~0.972 — essentially identical across the normal and skew conditions (the free `pi_c`
  nests the higher-order-implied distribution), 100% convergence with every replication
  finite and on the simplex. Compute lives in `mlsirm_core::cdm::fit_seq_gdina_qr`; exposed
  to Python as `fit_seq_gdina_qr` with the `SeqGdinaQrFit` wrapper (`item_step_prob` ragged
  accessor over the per-step layout). The shared-Q `fit_seq_gdina` is retained as the
  convenience special case.
- **Higher-order G-DINA** (de la Torre & Douglas, 2004; de la Torre, 2011).
  `fit_ho_gdina(responses, q_matrix)` fits the saturated G-DINA item model (each
  item's reduced attribute-mastery classes get a free success probability) under a
  *higher-order structural attribute prior*: a continuous trait `theta ~ N(0,1)`
  drives mastery, `P(alpha_k=1 | theta) = sigmoid(a_k theta + d_k)`, with attributes
  conditionally independent given the trait. It generalizes `fit_ho_cdm` (which
  restricts the item model to DINA slip/guess) and constrains `fit_gdina`'s free
  class distribution to the `2K`-parameter structured family. Estimated by
  marginal-ML EM over the joint `(alpha, theta)` grid: because the item response is
  conditionally independent of the trait given the attributes, the saturated item
  M-step `p_il = R_il/I_il` marginalizes the trait out exactly (reusing `fit_gdina`'s
  closed form), and the structural step is `K` independent 2PL calibrations of
  attribute mastery on the trait (reusing `fit_ho_cdm`'s Newton). The higher-order
  parameters are identified for `K >= 3`. Validated by a non-trivial anchor (a free
  saturated fit of DINA-patterned data recovers the DINA identity-link `delta`
  *and* the higher-order parameters), an independent-attribute pi-recovery check, and
  a 500-replication Monte-Carlo study (K=3, N=1500) — the saturated item
  probabilities recover with mass-weighted RMSE ~0.02 and attribute agreement > 0.9
  under both a normal and a skewed trait distribution. Extends `mlsirm_core::cdm`
  (reuses `reduce_class`, `mobius_inverse_inplace`, `newton_attr_2pl`,
  `ho_pi_from_params`). Exposed to Python through PyO3 as `fit_ho_gdina` with the
  `HoGdinaFit` wrapper.

- **Rating Scale Model** (Andrich, 1978). `fit_rsm(responses)` fits the Rasch-family
  polytomous model for items on a common rating scale (e.g. Likert): every item has
  its own location `delta_i`, but the `K-1` category thresholds `tau_k` are *shared
  across all items* — `ln[P(X=k)/P(X=k-1)] = theta - delta_i - tau_k`, `theta ~
  N(0,1)`. This is a constrained partial-credit model (the PCM/GPCM in `poly.rs` /
  `mixed.rs` have item-specific thresholds); at `K=2` it reduces exactly to the Rasch
  model. Implemented as the GPCM cell with slope 1 and the structured intercept
  `-k*delta_i - sum_{m<=k} tau_m` (reusing `poly::gpcm_logprobs`), fit by marginal-ML
  EM with a monotone ECM M-step: a per-item Newton for the locations, then a joint
  Newton for the shared thresholds aggregated over items — both with a backtracking
  line search that guarantees the marginal likelihood ascends — followed by
  re-centering the thresholds to sum to zero (the model is invariant under
  `tau -> tau - c`, `delta -> delta - c`). A 500-replication Monte-Carlo study (J=12,
  K=5, N=1000) recovers the item locations and the shared thresholds tightly and the
  trait with correlation > 0.85 under both a normal and a skewed trait distribution.
  New `mlsirm_core::rsm` module; exposed to Python through PyO3 as `fit_rsm` with the
  `RsmFit` wrapper.

- **Continuous Response Model** (Samejima, 1973) — the library's first estimator
  for a *continuous* bounded response (all other models are binary, polytomous,
  response-time, or cognitive-diagnosis). `fit_crm(responses)` fits Samejima's CRM,
  the limit of the graded response model as the number of ordered categories grows
  without bound. Operationally (Wang & Zeng, 1998), the logit of a response
  `Z in (0,1)` is conditionally normal and linear in the trait:
  `logit(Z_ij) | theta_j ~ N(a_i theta_j + d_i, sigma_i^2)`, `theta ~ N(0,1)`. The
  working `(slope a_i, intercept d_i, residual sd sigma_i)` map to the classic
  `(discrimination alpha_i = a_i/sigma_i, difficulty b_i = -d_i/a_i, scale
  gamma_i = a_i)`, all reported. Estimated by marginal-ML EM over a Gauss-Hermite
  trait grid with a **closed-form** weighted-least-squares item M-step (regress the
  transformed response on the trait under the posterior, then the residual
  variance) — the exact profile MLE, no Newton iteration. Continuous responses are
  information-rich, so a 500-replication Monte-Carlo study (J=15, N=500) recovers
  the item parameters tightly and the trait with correlation > 0.9 under both a
  normal and a skewed trait distribution. New `mlsirm_core::crm` module (reuses the
  `quadrature::gh_rule` grid); exposed to Python through PyO3 as `fit_crm` with the
  `CrmFit` wrapper. The `Z -> logit` Jacobian is a data-only constant, so the
  reported log-likelihood is in the transformed metric.

- **Higher-order structured attribute prior for cognitive diagnosis** (de la Torre
  & Douglas, 2004). `fit_ho_cdm(responses, q_matrix, model="dina"|"dino")` fits a
  DINA/DINO model whose `2^K` attribute-class distribution, instead of being free
  (as in `fit_cdm`), is *structured* by a continuous higher-order trait
  `theta ~ N(0,1)`: `P(alpha_k=1 | theta) = sigmoid(a_k theta + d_k)` with attributes
  conditionally independent given the trait. This replaces the `2^K - 1` free class
  probabilities with `2K` interpretable attribute parameters (slope `a_k`,
  intercept `d_k`). Estimated by marginal-ML EM over the joint `(alpha, theta)` grid:
  the item slip/guess M-step is unchanged, and the population update becomes `K`
  independent 2PL calibrations of attribute mastery on the trait (reusing the
  `fit_mmle_2pl` Newton with expected node counts). The implied class distribution,
  per-person trait EAP, MAP profile, and marginal attribute mastery are returned.
  The observed-data likelihood depends on `(a_k, d_k)` only through the implied class
  distribution, so the higher-order parameters are a genuine, identified restriction
  only for `K >= 3` (at `K <= 2` only the class distribution and the attribute
  classification are identified); `attr_slope` is anchored non-negative. A
  500-replication Monte-Carlo study (higher-order DINA, K=3, N=1000) recovers the
  attribute parameters and classification under both a correctly-specified normal
  trait and a mis-specified skewed trait. Extends `mlsirm_core::cdm` — reuses the
  DINA gate, `update_item`, and `mmle::GH_NODES`/`GH_WEIGHTS`. Exposed to Python
  through PyO3 as `fit_ho_cdm` with the `HoCdmFit` wrapper.

- **Item-level cognitive-diagnosis model selection by the Wald test** (de la
  Torre, 2011). `gdina_wald_selection(responses, q_matrix, alpha=0.05)` tests, for
  each item, whether the saturated G-DINA can be replaced by a more parsimonious
  reduced model. The candidates are exact *linear restrictions* of the
  identity-link parameters `delta = M^{-1} P` (`P` the reduced-class success
  probabilities): **DINA** (conjunctive — only the intercept and the top-order
  interaction free), **DINO** (disjunctive — the non-intercept coordinates tied
  onto one line `delta_S = (-1)^{|S|+1} Delta`, a general non-coordinate
  restriction), **A-CDM** (additive on the identity link — all interaction
  coordinates zero), **LLM** (linear logistic model — additive on the *logit* link),
  and **R-RUM** (reduced reparameterized unified model — additive on the *log* link).
  The Wald statistic `W = (R delta)' (R Sigma_delta R')^{-1} (R delta) ~ chi^2(df)`
  restricts the identity-link `delta = M^{-1} P` for DINA/DINO/A-CDM and the
  transformed `delta^h = M^{-1} h(P)` for LLM (`h = logit`) and R-RUM (`h = log`).
  For the identity link `Sigma_delta = M^{-1} Var(P) M^{-T}` with
  `Var(P_l) = P_l(1-P_l)/I_l`; for a transformed link the first-order delta method
  gives `Var(h(P_l)) = h'(P_l)^2 Var(P_l)` (LLM `1/(I_l P_l(1-P_l))`, R-RUM
  `(1-P_l)/(I_l P_l)`), sharing the same Möbius sandwich. All three covariances (and
  the two transformed deltas) accumulate in one pass over the shared Möbius columns
  `c_l = M^{-1} e_l` (reusing `mobius_inverse_inplace`); the expected reduced-class
  counts `I_l` come from one posterior pass. Per item the fewest-parameter model not
  rejected at `alpha` is selected (DINA and DINO cost two parameters; A-CDM, LLM and
  R-RUM each cost `1 + K`, so ties are broken by the larger p-value), else the
  saturated G-DINA. The covariance uses complete-data (expected) rather than
  observed information, so the test is mildly liberal — a 500-replication
  Monte-Carlo study (K=2, N=3000, strong attribute identification) confirms Type I
  error near nominal under both uniform and correlated/skew attribute distributions
  (Type I at `alpha=0.05`: DINA/DINO/A-CDM/LLM/R-RUM all within ~0.059–0.083) with
  power 0.98–1.000 against false over-restrictive or wrong-link models — including the
  cross-link cases (A-CDM and R-RUM rejected under LLM truth ~1.000/0.98, LLM rejected
  under R-RUM truth 1.000), verifying the link transform is faithful rather than
  cosmetic. A
  non-centered anchor test drives this home: truths additive on *only* one of the
  three links (identity/logit/log) are each recovered as their own model while the
  other two additive models are rejected. Extends `mlsirm_core::cdm` (reuses
  `fit_gdina`, `reduce_class`, `posterior_row_gdina`, `mobius_inverse_inplace`, and
  `fitstats::chi2_sf`). Exposed to Python through `gdina_wald_selection` /
  `WaldModelSelection` (both generic in the model count, so the two new candidates
  flow through unchanged). Deferred: the incomplete-data (observed-information)
  covariance.

- **Empirical Q-matrix validation by the PVAF method** (de la Torre & Chiu,
  2016). `validate_q_matrix(responses, provisional_q, epsilon=0.95)` checks and
  corrects the attribute-by-item Q-matrix of a cognitive-diagnosis model. Each
  candidate q-vector groups the `2^K` latent attribute classes into masters vs.
  non-masters of its required attributes; the *proportion of variance accounted
  for* is `PVAF(q) = zeta^2(q) / zeta^2_full`, the share of the item's
  across-class success-probability variance that grouping captures. Per item the
  method returns the q-vector with the **fewest** required attributes whose
  `PVAF >= epsilon` — an under-specified provisional q falls short and is
  enlarged, an over-specified one is trimmed because a smaller vector already
  clears the cutoff. The class weights and identified attribute labels come from
  a G-DINA fit under the provisional Q; each item's *saturated* success
  probability over all `2^K` classes is then recovered nonparametrically from
  the fitted posteriors, so a mis-specified item's true dependence is exposed by
  the attributes the *other* items identify (the method assumes the provisional
  Q is mostly correct). Extends `mlsirm_core::cdm` — reuses the G-DINA
  `reduce_class` collapse and posterior pass; the exhaustive q-vector search is
  `O(J * 4^K)`, so `K` is capped at 10. Validated by an anchor (the true Q
  validates to itself), over-/under-specification correction, and a
  500-replication Monte-Carlo Q-recovery study (K=3, J=15, N=1000): under a
  uniform attribute distribution the exact q-vector is recovered for 98.1% of
  items (attribute TPR 0.996, FPR 0.012), and under a correlated/skew
  higher-order distribution for 93.5% (TPR 0.982, FPR 0.035). Exposed to Python
  through PyO3 as `validate_q_matrix` with the `QMatrixValidation` wrapper.
  Deferred: the stepwise Wald item-level model-selection test (de la Torre,
  2011) and sequential/iterative Q-matrix re-estimation.

- **Testlet response model** (Bradlow, Wainer, & Wang, 1999; Wang, Bradlow, &
  Wainer, 2002). `fit_testlet(responses, testlet_id, model="rasch"|"2pl")` models the
  local dependence induced when items share a common stimulus (a reading passage): each
  item in testlet `d` carries a person-specific random effect `gamma_{j,d} ~ N(0,
  sigma^2_d)`, so `P(X=1) = sigmoid(a_i(theta_j - b_i - gamma_{j,d(i)}))`. The per-testlet
  variance `sigma^2_d` is the estimand of interest — a large value flags strong
  within-bundle dependence; all `sigma^2_d = 0` is the ordinary conditional-independence
  2PL/Rasch model. A dedicated estimator (not the general bifactor): because each item
  depends on `theta` and exactly one testlet effect, the marginal likelihood **factors**
  into a `theta`-outer / per-testlet-`gamma`-inner nested Gauss-Hermite quadrature whose
  per-person cost is independent of the number of testlets `D` (vs the bifactor's
  exponential `D`-dimensional grid), and it reports `sigma^2_d` directly rather than only
  per-item loadings. The item M-step reuses `fit_mmle_2pl`'s Newton on the effective node
  `t_g - sigma_d*u_h`; the closed-form variance update `sigma^2_d <- sigma^2_d * mean_j
  E[u_d^2 | y_j]` is accelerated with SQUAREM (Varadhan & Roland, 2008; monotone, with a
  plain-EM fallback) to tame the slow variance-component convergence. Singleton testlets
  (whose variance is non-identified) are pinned to 0. Compute lives in
  `mlsirm_core::testlet::fit_testlet`; the shared Newton and Gauss-Hermite table make the
  `sigma^2 -> 0` case reduce **bit-exactly** to `fit_mmle_2pl` (the reduction anchor,
  asserted `< 1e-12`). Also anchored: a no-spurious-LD check (pure-2PL data recovers
  `sigma^2 ~ 0`), a strong-LD recovery with a log-likelihood gain over the naive 2PL fit,
  singleton pinning, and a monotone-ascent guard. A Bradlow-Wainer-Wang-style
  500-replication Monte-Carlo (Rasch testlet, N=1000, D=4) under normal and skewed
  ability recovers the testlet variances near-unbiasedly (RMSE ~0.093, `|bias| <= 0.007`)
  and the item difficulties (RMSE ~0.09), with every replication converging. Exposed via
  PyO3 as `fit_testlet` with the
  `TestletFit` Python wrapper. (In the 2PL testlet the discrimination `a_i` and the
  testlet SD `sigma_d` both scale the dependence via `a_i*sigma_d` and separate only
  weakly, so the Rasch testlet is the well-identified default.) Deferred: polytomous and
  3PL testlets, covariate/second-order structure, and the original paper's fully-Bayesian
  MCMC estimator.

- **Linear Logistic Test Model (LLTM)** (Fischer, 1973). An *explanatory* Rasch
  model: `fit_lltm(responses, q_design)` decomposes each item's easiness (the package's
  additive sign convention; Fischer difficulty is its negative) into a
  linear combination of `K` basic cognitive-operation parameters through a fixed
  weight matrix `Q` (`b_i = c + Σ_k q_ik·η_k`), rather than estimating `J` free item
  easinesses. With `K << J` parameters it tests whether a small set of cognitive
  operations *explains* the item parameters. Estimated by marginal-ML EM: the
  E-step is the Rasch node posterior over the shared Gauss-Hermite rule; the M-step is
  a `K`-dimensional chain-rule Newton — the per-item Rasch easiness gradient/Hessian
  aggregated through the design (`g_η = Qᵀg_b`, `H_η = Qᵀ diag(h_b) Q + ridge`, solved
  with the shared dense `solve_small`). A free grand-mean easiness intercept is fit by
  default. The classic likelihood-ratio test of LLTM vs the saturated Rasch model
  (`2·(ll_Rasch − ll_LLTM) ~ χ²(J − K − 1)`) is computed inline (the Rasch reference is
  the same engine run with `Q = I`). **Identification is validated, not assumed**: the
  effective design (including the intercept column) must have full column rank for `η`
  to be identified, so a rank-deficient `Q` (e.g. one whose rows sum to a constant,
  colliding with the intercept) is rejected rather than papered over by the Newton
  ridge. Compute lives in `mlsirm_core::lltm::fit_lltm`; because the M-step reuses
  `mmle`'s Rasch Newton and Gauss-Hermite table, the `Q = I` case reduces
  **bit-exactly** to a Rasch fit — anchored two ways: a single M-step is bit-identical
  (`==`) to `J` independent per-item Rasch Newton steps, and a full `Q = I` fit matches
  a single-class Rasch mixture fit to `< 1e-10`. A 500-replication Monte-Carlo
  (J=30, K=5, N=1500) under normal and skewed ability recovers the basic parameters
  (RMSE/bias) and induced easinesses, and validates the LR test (Type I when the
  restriction holds, power when it is violated off-model). Exposed via PyO3 as
  `fit_lltm` with the `LltmFit` Python wrapper. This is the marginal-ML / `N(0,1)`
  operationalization of Fischer's conditional-ML LLTM. It is a repository-specific
  estimator choice, and finite-sample equality with Fischer's conditional-ML item
  estimates is not asserted. Deferred: conditional-ML estimation, LLTM for 2PL/polytomous
  models, and random-weights / LLRA extensions.

- **Mixed Rasch / mixture IRT** (Rost, 1990; Rost & von Davier, 1995). A new
  paradigm for unobserved population heterogeneity: `fit_mixture(responses,
  n_classes, model="rasch"|"2pl")` models the population as a mixture of `C` latent
  classes, each with its OWN item parameters and a mixing weight `pi_c`, detecting
  qualitatively different response strategies a single-class model cannot represent.
  Within a class, responses follow a Rasch (discrimination fixed at 1) or 2PL model
  with `theta ~ N(0,1)`, estimated by marginal-ML EM: the E-step forms the joint
  posterior over (class, ability node) via one max-shift log-sum-exp over the `C·Q`
  Gauss-Hermite grid; the per-class item M-step reuses the exact penalized Newton
  step of `fit_mmle_2pl` (weighted by the class responsibility); the mixing weights
  update to the mean posterior class membership. Because the mixture likelihood is
  multimodal, `n_starts > 1` runs random restarts (start 0 is a deterministic warm
  start) and keeps the highest-likelihood fit; classes are returned in a canonical
  order (mixing weight descending, ties by mean difficulty ascending) to tame label
  switching. Compute lives in `mlsirm_core::mixture::fit_mixture`; the shared Newton /
  Gauss-Hermite table with `fit_mmle_2pl` makes the `C = 1` case reduce **bit-exactly**
  to the verified single-class 2PL estimator — the reduction anchor, asserted to
  `< 1e-12`. Also anchored: a two-class difficulty-reversal recovery (the canonical
  Rost two-strategy structure), permutation-matched, plus a monotone-ascent guard. A
  500-replication Monte-Carlo (C=2, J=15, N=1500, reversal truth) under normal and
  skewed ability recovers the class difficulties (permutation-matched RMSE), mixing
  proportions, and class membership (MAP accuracy + label-invariant Adjusted Rand
  Index; Hubert & Arabie, 1985). Exposed via PyO3 as `fit_mixture` with the
  `MixtureFit` Python wrapper. This repository combines Rost's latent-class structure
  with a fixed-standard-normal, Bock-Aitkin marginal-ML EM estimator. It differs from
  the conditional-ML estimators in Rost (1990) and psychomix (Frick et al., 2012), so
  no exact finite-sample item-contrast equivalence is claimed. Deferred: free per-class
  ability variance, automatic model selection
  over `C` (AIC/BIC/ICL from the returned `n_parameters`/`loglik_trace`), and
  concomitant-variable mixing.

- **Generalized DINA (G-DINA), the saturated cognitive-diagnosis framework**
  (de la Torre, 2011). `fit_gdina(responses, q_matrix)` fits the general model of
  which DINA, DINO, A-CDM, LLM, and R-RUM are constrained special cases. For an
  item requiring `K_i` attributes, each of its `2^{K_i}` *reduced* attribute-mastery
  classes gets a **free** success probability `p_il = P(X_i = 1 | reduced class l)`,
  estimated by marginal-ML EM over the `2^K` profiles. The E-step reuses the DINA
  profile-grid posterior; the closed-form saturated M-step is
  `p_il = R_il / I_il` (expected correct / expected total in reduced class `l`) —
  exactly DINA's two-cell slip/guess step generalized to `2^{K_i}` cells (de la
  Torre, 2011, Eq. 10). The identity-link parameters `item_delta` (intercept, main
  effects, all interactions) are recovered from the fitted probabilities by an
  in-place signed subset Möbius transform `delta = M^{-1} p` (no matrix inverse), so
  the constrained submodels are readable off the `delta` pattern — DINA leaves only
  the intercept and the highest-order interaction nonzero; A-CDM zeroes the
  interactions. Item parameters are stored ragged (CSR: `item_off` + flat
  `item_prob`/`item_delta`) since `2^{K_i}` varies per item; the box constraint
  `0 <= p_il <= 1` holds for free (`0 <= R_il <= I_il`). The saturated estimator is
  otherwise order-unconstrained: Q-matrix identifiability does not make the
  all-mastered class largest, and the separate Hong, Chang, and Tsai (2016)
  subset-lattice order restriction is not implemented.
  Compute lives in `mlsirm_core::cdm::fit_gdina`, extending the DINA module without
  touching the shipped DINA core; exposed via PyO3 as `fit_gdina` with the `GdinaFit`
  Python wrapper. Correctness is anchored by a brute-force likelihood identity
  (log-space path == naive enumeration to `1e-12`), a **DINA-reduction crux anchor**
  (DINA-generated data recovers `p_il = g_i` for every non-top class and `1 - s_i`
  at the top, with the exact DINA `delta` pattern), a DINO-reduction anchor, an
  A-CDM additivity anchor (fitted interactions negligible relative to main effects),
  a Möbius round-trip identity, an exhaustive `reduce_class` bit-packing check, and a
  deterministic limit. A de la Torre (2011)-style 500-replication Monte-Carlo (K=5,
  J=30, N=1000) with a stochastic higher-order attribute distribution (de la Torre &
  Douglas, 2004) under normal and skewed abilities recovers `p_il` (mass-weighted
  RMSE) and attribute classification accuracy. Deferred: LLM/R-RUM logit/log-link
  submodels, item-level model-selection Wald tests, Q-matrix validation, and full
  subset-lattice isotonic monotonicity (Hong et al., 2016).

- **Cognitive diagnosis models: DINA and DINO** (Junker & Sijtsma, 2001; de la
  Torre, 2009; Templin & Henson, 2006). A new discrete-attribute paradigm
  alongside the continuous-trait family: `fit_cdm(responses, q_matrix,
  model="dina"|"dino")` classifies each respondent's binary attribute-mastery
  profile `alpha in {0,1}^K` against a Q-matrix of item-attribute requirements.
  The ideal response is the conjunctive AND gate `eta = prod_k alpha_k^{q_k}`
  (DINA — mastery of all required attributes) or the disjunctive OR gate
  `eta = 1 - prod_k (1-alpha_k)^{q_k}` (DINO — any required attribute), and the
  observed response adds a per-item slip `s_i = P(X=0|mastered)` and guess
  `g_i = P(X=1|not mastered)`, `P(X=1|alpha) = (1-s_i)^{eta}(g_i)^{1-eta}`.
  Estimation is marginal-ML EM over the `2^K` profiles with a free profile
  distribution: the E-step posterior is accumulated over the discrete profile
  grid (a bitwise gate test replaces the continuous quadrature), the item M-step
  is **closed form** (`s_i = 1 - R1_i/I1_i` = expected fraction of masters
  answering wrong; `g_i = R0_i/I0_i` = non-masters answering right; de la Torre,
  2009, Eqs. 9-10), and the population step is a mean of the posteriors. The
  monotonicity/identification constraint `1 - s_i > g_i` is enforced by the exact
  constrained boundary maximiser; missing cells are dropped under MAR. Persons
  are classified by the posterior-mode profile (`map_profile`) and marginal
  attribute-mastery probabilities (`attr_prob`, attribute EAP). All compute runs
  in the Rust core (`mlsirm_core::cdm::fit_cdm`) with the `2^K` profile grid
  bit-encoded (no `N*L` storage; streaming E-step); DINA and DINO share one
  estimator differing only in the one-line gate mask. Correctness is anchored by
  a brute-force likelihood identity (log-space path == naive enumeration to
  `1e-12`), a deterministic `s=g=0` limit (exact pattern recovery), a
  DINA==DINO gate-equivalence identity on single-attribute items, and a K=1
  reduction to a 2-class latent-class model. A de la Torre (2009)-style
  500-replication Monte-Carlo (K=5, J=30, N=1000) recovers slip/guess with mean
  RMSE 0.013-0.024 and negligible bias (`|bias| < 3e-4`) and attains attribute
  classification agreement 0.99 (s=g=0.1) / 0.95 (s=g=0.2), pattern-wise 0.96 /
  0.76. Deferred: the general G-DINA/saturated CDM, Q-matrix estimation, and
  structured (higher-order) attribute priors.

- **Polytomous response models (GRM / GPCM), unidimensional.** A complete
  fit -> score -> information subsystem: `fit_polytomous(responses, n_cat,
  model="grm"|"gpcm")` fits the graded response model (Samejima; the default)
  or the generalized partial credit model (Muraki) by Bock-Aitkin marginal-EM;
  `score_polytomous(responses, fit)` returns EAP trait scores and posterior
  SDs; `information_polytomous(fit, theta)` returns item and test Fisher
  information curves. `NaN` responses are treated as missing and marginalized
  out of each person's likelihood and posterior. All numerical work — the category cells, the residual
  M-step gradient, the Newton item update, the EAP reduction, and the
  information — runs in the Rust core (`mlsirm_core::poly`:
  `grm_logprobs`/`gpcm_logprobs` + `*_node_gradient` + `fit_poly_unidim` +
  `score_poly_eap` + `poly_item_information`), exposed via PyO3; the NumPy
  `category_logprobs`/`grm_category_logprobs`/`gpcm_node_gradient`/
  `fit_gpcm_numpy` are parity references held to `<= 1e-12` (both cells) /
  recovery agreement (fitter). GRM is
  chosen as the identification-clean default for the latent-space family — the
  single interaction term enters every cumulative logit as a shared shift, with
  no forced category scaling (design rationale and literature basis in
  `docs/papers/gpcm-nominal-design-spec.md`). The latent-space polytomous
  extension (the same cell inside the marginal `(theta, xi)` quadrature) is the
  next milestone.

- **Polytomous computerized adaptive testing** (Dodd, De Ayala & Koch, 1995).
  `cat_simulate_polytomous(true_theta, fit)` simulates an adaptive test over a
  fitted GRM/GPCM bank: items are selected by maximum Fisher information at the
  running EAP trait, responses are generated at the true trait, and the trait +
  posterior SD are re-estimated after each item, stopping at an SE threshold (or
  a fixed length). Returns per-simulee `theta_eap`, `theta_sd`, and `n_used`.
  Compute in Rust (`mlsirm_core::poly::poly_cat_simulate`, plus
  `poly_cat_next_item`), composing the existing item information and EAP scoring.
  Validated by a 500-simulee Monte-Carlo: a variable-length CAT recovers the
  trait to RMSE 0.29 (normal) / 0.33 (skew) using ~9.7 of 40 bank items, and at
  a fixed length of 12 maximum-information selection beats random (RMSE 0.27 vs
  0.33 normal; 0.30 vs 0.40 skew).

- **Polytomous person fit** (Drasgow, Levine & Williams, 1985; Snijders, 2001).
  `person_fit_polytomous(responses, fit)` returns the standardized
  log-likelihood `l_z` and its estimated-trait correction `l_z*` (per person,
  at the EAP trait) plus `theta_eap` and a boolean `flagged`, for a fitted
  GRM/GPCM — the ordered-category generalization of the binary l_z. Compute in
  Rust (`mlsirm_core::poly::poly_person_fit`), reusing the poly cells with a
  central-difference trait score. Validated by an exact reduction to the binary
  `person_fit` l_z at `n_cat = 2` (`<1e-6`) and a 500-replication Monte-Carlo:
  under model respondents `l_z*` is ~N(0,1) (mean −0.15, sd 1.04, Type I 0.08
  at a 20-item test), and inconsistent responders are flagged with power 0.86.

- **Nominal categories model** (Bock, 1972; Thissen, Cai & Bock, 2010).
  `fit_nominal_polytomous(responses, n_cat)` fits the unidimensional nominal
  model `P(Y=k|θ) = softmax_k(a_k·θ + c_k)` with a free scoring function `a_k`
  and intercept `c_k` per category, identified by `a_0 = c_0 = 0` and
  `θ ~ N(0,1)`, returning a `NominalFit` (`scores`, `intercepts`, `loglik`).
  The generalized partial credit model is the special case `a_k = a·k`, so the
  nominal model nests it. Compute in Rust (`mlsirm_core::poly::fit_nominal`),
  reusing the softmax cell and its residual gradient. The parameterization and
  identification were adversarially verified against the source chapter.
  Validated by the GPCM nesting (loglik ≥ the GPCM fit, recovered scores linear
  in `k`) and a 500-replication recovery Monte-Carlo (per-item sign alignment):
  under a matched `N(0,1)` ability the score RMSE is 0.15 with |bias| 0.01
  (near-unbiased), degrading to RMSE 0.44 / |bias| 0.39 under a skewed
  population.

- **Polytomous item-pair local dependence** (Chen & Thissen, 1997; Liu &
  Maydeu-Olivares, 2013). `local_dependence_polytomous(responses, fit)` returns,
  for every item pair of a fitted GRM/GPCM, the Pearson `X²` and likelihood-ratio
  `G²` comparing the observed `K×K` contingency table to the model-implied joint
  under local independence, with `df = (K-1)²`, the χ² p-value, Cramér's V, and
  the largest standardized cell residual — the ordered-category generalization
  of the binary pairwise χ² and the pair-level complement to item-level S-X² and
  test-level M2. Compute in Rust (`mlsirm_core::fitstats::poly_local_dependence`).
  Validated by a deterministic K=2 reduction to a from-scratch 2×2 χ² and a
  500-replication Monte-Carlo at fitted parameters: locally-independent pairs are
  calibrated (X²/df = 0.84, Type I 0.03 — conservative, as the papers note),
  while an injected 2-item testlet is localized to that pair (X²/df = 10.9, power
  1.00).

- **Polytomous IRT likelihood-ratio DIF** (Thissen, Steinberg & Wainer, 1993;
  Woehr & Meriac, 2010). `dif_polytomous(responses, group_id, n_cat)` runs a
  two-group DIF sweep for GRM/GPCM items: it fits a *compact* model (all items
  group-invariant) once, then per studied item an *augmented* model (that item's
  parameters freed per group) with every other item as the anchor, and refers
  `LR = 2·Δloglik` to `χ²((n_groups−1)·n_cat)`. Each non-reference group's latent
  distribution `N(μ_g, σ_g²)` is estimated in **both** models (group 0 pinned to
  `N(0,1)`), so genuine ability differences between groups (impact) are absorbed
  rather than mistaken for DIF. Returns per-item `lr`, `df`, `p_value`,
  `flagged_bh` (Benjamini-Hochberg FDR), and `effect_size` (the across-group
  range of the item's mean category location). Compute in Rust
  (`mlsirm_core::poly::fit_poly_multigroup` — a Bock-Zimowski multi-group
  marginal EM whose per-item M-step reuses the single-group Newton step on each
  group's nodes/expected-counts stacked, the concatenation being exactly the
  Bock-Zimowski pooling — driving `poly_dif_sweep`). Validated by a 500-rep
  Monte-Carlo with impact (focal `θ~N(0.5, 1.2²)`), two-group GPCM, `K=3`:
  under no DIF the test is calibrated (Type I 0.042, `mean(LR)=2.92≈df=3`), an
  injected uniform difficulty shift is detected with power 0.996 and a
  non-uniform slope difference with power 0.920, while a skewed focal population
  inflates Type I only mildly (0.057); a structural check confirms the augmented
  fit never falls below the compact one and recovers the focal `μ, σ`.

- **Response-time person fit** (van der Linden & Guo, 2008; Sinharay, 2018).
  `rt_person_fit` flags aberrant response-time patterns — rapid guessing, item
  preknowledge — under a fitted lognormal RT model. It profiles each person's speed
  by ML, so the sum of squared standardized log-time residuals
  `W_j = sum_i [alpha_i (ln T_ij - (beta_i - tau_hat_j))]^2` is *exactly*
  `chi2(n_j - 1)` (an orthogonal-projection identity — the estimated-speed
  correction is a clean loss of one degree of freedom, the RT analogue of `l_z*`,
  with no asymptotic drift). It returns the aggregate `W`/p-value, a Wilson-Hilferty
  standardized `l_t`, and per-item studentized residuals plus one-sided too-fast
  flags. It detects speed *inconsistency across items*, not a uniform speed level
  (the profile absorbs it). Compute in Rust (`rt::rt_person_fit`, reusing
  `fitstats::chi2_sf`); exposed via PyO3 and Python. Validated by an exact identity
  anchor (at true parameters the residuals are `N(0,1)` and `W` is `chi2(n)` with
  known speed, `chi2(n-1)` once profiled, to within Monte-Carlo error) and a
  500-replication Monte-Carlo: Type I sits on nominal (0.05, exact — no
  finite-length conservatism), rapid-guessing and preknowledge responders are
  detected with power ~1.0 under both normal and skew speed, the flag is robust to
  the speed-distribution shape (it conditions on within-item residuals), and the
  tampered items are recalled at ~99%. Deferred: an EAP-plug-in mode (statistically
  inferior — it mis-calibrates the chi-square) and multivariate RT aberrance.

- **Joint speed-accuracy hierarchical model** (van der Linden, 2007, Level 2). A
  new `mlsirm_core::rt_joint` module and the public `fit_speed_accuracy` — the
  person-level layer that ties ability `theta` (from an accuracy 2PL model) to
  speed `tau` (from the lognormal RT model) through a bivariate-normal person
  distribution `(theta, tau) ~ N2(0, [[1, rho*sigma_tau], [rho*sigma_tau,
  sigma_tau^2]])`, with the accuracy responses and log-times conditionally
  independent given `(theta, tau)`. The headline output is `rho`, the ability-speed
  correlation. This is the two-stage estimator: item parameters are held fixed and
  the person covariance `(rho, sigma_tau)` is estimated by marginal ML over a 2-D
  Gauss-Hermite grid built by Cholesky-mapping the standard nodes through
  `Sigma_P`, with an exact constrained EM M-step (`c = S12/S11`,
  `v = S22 - S12^2(S11-1)/S11^2`). The reported `rho` is the consistent marginal-ML
  correlation, not the shrinkage-attenuated correlation of the two separate EAPs.
  Compute in Rust (`rt_joint::fit_speed_accuracy_covariance`); exposed via PyO3 and
  Python. Validated by an exact identity anchor (at `rho = 0` the 2-D grid
  log-likelihood factorizes into the sum of the two 1-D grids to `< 1e-10`), a
  reduction anchor (true independence returns `rho ~ 0`), monotone EM, and a
  500-replication Monte-Carlo recovering `rho in {0, 0.5, -0.5}` with essentially
  zero bias (bias `< 0.001`, RMSE ~0.03-0.04) and `sigma_tau` to RMSE ~0.008.
  Deferred: the one-step full-information MMLE, 3PL guessing, and item-parameter-
  uncertainty propagation into SE(rho).

- **Lognormal response-time model** (van der Linden, 2007). A new
  `mlsirm_core::rt` module and the public `fit_response_times` — the speed-side
  analogue of the 2PL for item response *times*, opening a response-time modality
  alongside the accuracy models. For person `j` (latent speed `tau_j`) and item
  `i` (time intensity `beta_i`, time discrimination `alpha_i`),
  `ln(T_ij) ~ Normal(beta_i - tau_j, 1/alpha_i^2)`; item parameters and the speed
  SD are estimated by marginal-ML EM with `tau ~ Normal(0, sigma_tau^2)`, and speed
  is scored by EAP. Because the model is conditionally Gaussian with a unit loading
  on `tau`, the speed posterior, marginal likelihood, and EAP are all *exact closed
  forms* (matrix-determinant / Sherman-Morrison), so the estimator needs neither
  quadrature nor a line search — the EM is exact `O(nnz)` coordinate ascent. The
  log-time metric identifies the speed scale (so `sigma_tau` is estimated, not
  fixed) and only the location is pinned (`mu_tau = 0`). Compute in Rust; exposed
  via PyO3 and Python; missing/non-positive times are marginalized per person.
  Validated by an exact identity anchor (the closed-form marginal log-likelihood
  equals a dense multivariate-normal log-pdf to `< 1e-9`), a reduction anchor
  (`sigma_tau -> 0` collapses to the per-item lognormal MLE), and a 500-replication
  Monte-Carlo: under both normal and a *misspecified* skew speed population the item
  parameters stay essentially unbiased (RMSE `alpha` 0.067 / `beta` 0.027, bias
  `beta` -0.0001 under skew) with speed recovered at corr 0.92, demonstrating that
  the level-1 RT item parameters are estimable independently of the speed
  distribution's shape. Deferred: the joint speed-accuracy hierarchical layer,
  Louis-standard-error information, and RT bank linking.

- **Standard errors of equating** (Kolen & Brennan, 2014, ch. 7; Efron &
  Tibshirani, 1993). `equating_standard_errors` reports the per-score-point
  sampling error of the equated score for the equivalent-groups design, by two
  routes. The nonparametric **bootstrap** (`route="bootstrap"`) resamples
  examinees per group independently with replacement at the observed sample sizes,
  re-equates each of `n_boot` replicates through the existing equating code, and
  returns the per-score bootstrap SD and a percentile confidence interval — it
  works for every method including equipercentile, which has no simple analytic
  SEE. The **delta-method** (`route="analytic"`) returns the closed-form
  normal-theory SE for mean equating (`sigma_x^2/n_x + sigma_y^2/n_y`, constant in
  `x`) and linear equating (`sigma_y^2 (1 + z^2/2)(1/n_x + 1/n_y)`,
  `z = (x-mu_x)/sigma_x`). Compute in Rust (`equating::bootstrap_see` /
  `analytic_see`); exposed via PyO3 and Python. Validated by the analytic-Linear
  agreeing with the bootstrap-Linear SEE within Monte-Carlo tolerance, the Mean
  SEE being constant, a `1/sqrt(N)` shrink and seed-determinism check, and a
  500-replication Monte-Carlo confirming the bootstrap SE recovers the *true*
  sampling SD of `e_Y(x)` (from an outer fresh-sample Monte-Carlo) — interior
  ratio in [0.95, 1.08] for equipercentile. Deferred: NEAT bootstrap SEE, analytic
  equipercentile/kernel SEE.

- **Tucker & Levine linear NEAT equating** (Kolen & Brennan, 2014, §4.3–4.4;
  Brennan, 2006). `equate_neat_linear` adds the linear observed-score methods for
  the common-item non-equivalent-groups design, alongside the existing chained /
  frequency-estimation equipercentile NEAT. Each forms synthetic-population
  moments of the two forms (weighted by `w1`) from a group total-on-anchor slope
  `gamma` — Tucker uses the regression slope `Cov(total, V)/Var(V)`; Levine uses
  the congeneric effective-length ratio, which differs for an internal anchor
  (`Var(total)/Cov`) versus an external one (`(Var(total)+Cov)/(Var(V)+Cov)`) —
  then equates linearly. Compute in Rust (`equating::equate_neat_linear`); exposed
  via PyO3 and Python. Validated by the exact reduction to equivalent-groups
  linear equating under equal anchor moments (all four Tucker/Levine ×
  internal/external variants, any `w1`, to `< 1e-9`), a hand-computed check that
  pins the internal-vs-external Levine gamma against an independent oracle, and a
  500-replication Monte-Carlo under a common-regression generative model
  (equated-score interior RMSE 0.39 → 0.19 from `N = 1000` to `4000`, ratio 2.02 ≈
  √4; max bias 0.051 → 0.034). Deferred: Levine true-score equating, Braun-Holland.

- **Kernel equating + log-linear presmoothing** (von Davier, Holland & Thayer,
  2004; Holland & Thayer, 2000). Two enhancements to the equating module.
  `loglinear_smooth(counts, degree)` presmooths a score-frequency distribution by
  Poisson-ML log-linear fitting (on an orthonormal polynomial design over a
  centered/scaled score, Newton with step-halving), preserving the first `degree`
  sample moments exactly while damping sampling noise; it returns AIC/BIC so a
  caller can select the degree, and saturated at `degree = k` it reproduces the
  raw relative frequencies. `equate_observed_scores_kernel` adds a Gaussian-kernel
  continuization (von Davier's `F_h(x) = Σ_j r_j Φ((x − a x_j − (1−a)μ)/(a h))`,
  bandwidth by the penalty method) and optional per-form presmoothing to the
  equipercentile family, behind a single extended entry point whose uniform-kernel
  path reproduces the existing equipercentile bit-for-bit. Compute in Rust
  (`equating::loglinear_smooth` / `equate_eg_ext`); exposed via PyO3 and Python.
  Validated by exact-identity anchors — uniform-kernel equating equals the
  equipercentile to `< 1e-12`; presmoothing preserves the first `T` moments to
  `< 1e-8` and reproduces `rel_freq` when saturated; the Gaussian-kernel
  self-equate is the identity, a large bandwidth drives kernel equating to linear
  to `< 1e-4`, and the continuized density preserves the discrete mean and
  variance — plus a 500-replication Monte-Carlo against the population
  Gaussian-kernel transform (interior RMSE 0.53 → 0.26 from `N = 1000` to `4000`,
  ratio 2.03 ≈ √4; max bias 0.049 → 0.020). Deferred: bivariate presmoothing,
  kernel-NEAT, and analytic standard errors.

- **Observed-score equating** (Kolen & Brennan, 2014). A new
  `mlsirm_core::equating` module and the public `equate_observed_scores` /
  `equate_neat` — the raw-score complement to the IRT scale linking (`irt_link`).
  Equivalent-groups mean, linear, and equipercentile equating (percentile-rank
  matching with the Kolen-Brennan uniform-kernel continuization, equated scores
  kept real-valued), and the common-item non-equivalent-groups (NEAT) design via
  chained equipercentile and frequency-estimation (post-stratification)
  equipercentile. The attainable min/max are computed on relative-frequency
  vectors; the frequency-estimation synthetic densities are renormalized so a
  poorly overlapping anchor degrades toward each group's own marginal rather than
  corrupting the cdf. Compute in Rust; exposed via PyO3 and a Python
  `equating.py` (`EquateResult`). Validated by three exact identities — the
  equipercentile self-equate is the identity to `< 1e-9` (including the low
  boundary at `x = 0`), mean/linear recover a known integer-affine transform to
  `< 1e-9`, and both NEAT methods collapse to EG equipercentile under equal
  anchor distributions to `< 1e-9` — plus a 500-replication Monte-Carlo against a
  deterministic Lord-Wingersky population equating: the empirical equipercentile
  converges at the expected rate (interior RMSE 0.53 at `N = 1000` → 0.26 at
  `N = 4000`, ratio 1.99 ≈ √4; max bias 0.068 → 0.031). Deferred (each a drop-in
  behind the density/table interface): Tucker/Levine linear NEAT, log-linear
  presmoothing, and Gaussian-kernel equating (von Davier et al., 2004).

- **Nonparametric polytomous person fit U3poly** (Emons, 2008; van der Flier,
  1982). `u3_person_fit_polytomous(responses, n_cat)` computes van der Flier's
  `U3` person-fit statistic generalized to ordered polytomous items — a
  *model-free* index: each item-step response function `P(Y_i >= m)` is estimated
  by its sample proportion, turned into a logit weight, and a person's observed
  weighted score is compared to the largest and smallest weighted scores
  attainable at that person's total score (the conditioning group), giving
  `U3 in [0, 1]` (1 = maximally popularity-inconsistent). The attainable min/max
  bounds are computed by exact min-plus / max-plus DP (not the flat "sum of the
  top-k weights" shortcut, which over-counts once an unused category breaks
  within-item monotonicity). `u3_cutoff_polytomous(fit, n_persons)` returns a
  simulated `1 - alpha` critical value by parametric bootstrap (U3poly has no
  usable analytic null; Emons used simulated critical values). Compute in Rust
  (`mlsirm_core::poly::u3_poly_person_fit` + `u3_poly_bootstrap_cutoff`).
  Validated by an exact `n_cat = 2` reduction to a from-scratch van der Flier `U3`
  (max abs diff `< 1e-10`) and a 500-replication Monte-Carlo (GPCM, `K = 5`,
  `n = 600`): the simulated cutoff calibrates the marginal flag rate under a
  matched population (Type I 0.052 normal / 0.054 skew) and detects careless
  responders with power ~1.00; the per-total-score-group flag-rate deviation
  (0.066 normal / 0.083 skew) is reported to make transparent that a single
  pooled cutoff cannot fully condition on the total score. Complements the
  parametric `l_z`/`l_z*` (`person_fit_polytomous`) with a distribution-free
  screen.

- **Polytomous M2 limited-information goodness-of-fit** (Maydeu-Olivares & Joe,
  2014). `m2_polytomous(responses, fit)` returns the test-level M2 statistic,
  `df`, `p_value`, RMSEA2 (with a 90% interval), and SRMSR for a fitted GRM/GPCM
  — the ordered-category generalization of the binary M2 (`m2_stat`). It uses
  the cumulative marginals `P(Y_i>=c)` and `P(Y_i>=c, Y_j>=d)` (the same M2 as
  the paper's category-equality form) and reduces **exactly** to the binary
  `m2_rmsea2` at `n_cat = 2`. Compute in Rust (`mlsirm_core::fitstats::poly_m2`),
  reusing the one-Cholesky residual-projection solve. `df = n(K-1) +
  C(n,2)(K-1)² - nK`. Validated by the exact `K=2` reduction (GRM and GPCM) and
  a 500-replication Monte-Carlo: under a matched `N(0,1)` ability `mean(M2)/df =
  0.99` with Type I error 0.05 (nominal), and under a skewed population `M2`
  inflates 4× with power 1.00.

- **Generalized S-X² item fit for polytomous models** (Kang & Chen, 2008, 2011).
  `item_fit_polytomous(responses, fit)` returns the per-item summed-score
  chi-square, `df`, `p_value`, and retained cell count for a fitted GRM/GPCM,
  extending the binary Orlando-Thissen S-X²: persons are grouped by summed
  score, and the model-expected category proportions come from the generalized
  Lord-Wingersky recursion (Thissen, Pommerich, Billeaud & Williams, 1995) with
  the leave-one-out summed-score distribution. Boundary score groups are merged
  and adjacent categories collapsed to a minimum expected frequency. Compute in
  Rust (`mlsirm_core::poly::poly_s_x2`), exposed via PyO3. Validated to reduce
  **exactly** to the trusted binary `fitstats::s_x2` at `n_cat = 2` (GRM and
  GPCM, statistic and df), and — at the true generating parameters — to track
  its reference chi-square (`E[S-X²] ≈ Σ cells`) for both the GPCM (2008) and
  GRM (2011) families.

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
- **GPU EAP scoring kernel** (`mlsirm_core::gpu_marginal::score_eap_gpu`, WGSL
  `score_pass`): Bock-Mislevy (1982) EAP scoring on the wgpu path, one thread
  per person (race-free — each person owns its output slots, unlike the E-step
  reduction), reusing the same `cell_l` binary-sparsity table decomposition.
  Exposed as an **opt-in** device on `score_eap_device(..., Device::Gpu)` and
  through PyO3 `score_bank_eap(..., device=...)` and
  `serving.score_respondents(..., device="gpu")`; the default stays the exact
  f64 CPU reduction, so precision-sensitive paths and serving parity are
  unchanged. f32 kernel, GPU-vs-CPU parity ≤ 2e-3 verified on-device
  (`gpu_eap_matches_cpu_reduction`); falls back to CPU with no adapter or when
  `n_dims`/`latent_dim > 8`. Extends GPU offload from the E-step to the 31k-
  person serving hot path (project compute policy: all math in Rust, GPU where
  it pays).
- **IRT scale linking for common-item designs** (`fast_mlsirm.irt_link`;
  `mlsirm_core::linking`): the moment methods (mean/mean, mean/sigma) and the
  characteristic-curve methods of Haebara (1980) and Stocking & Lord (1983) for
  putting a separately-calibrated new form onto the reference scale
  (`theta_old = A·theta_new + B`), motivated by the mixed-format / multi-study
  linking papers in the corpus (Kim & Lee 2006; Yao & Boughton 2009; Brossman &
  Lee 2013). The characteristic-curve loss is minimized by a self-contained
  Nelder-Mead over `(A, B)` from the mean/sigma start, integrated over a
  standard-normal Gauss-Hermite grid. Rust compute path; recovery tests for all
  four methods in both suites. (Complements the existing anchor-based
  `link_fixed_item_parameters` and the FIPC serving path.)
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

### Added

#### Rust-only literature true-parameter recovery gate

- A Rust-only true-parameter recovery experiment for a bounded representative
  Kang and Jeon (2025) MLS2PLM simulation cell (`P = 500`), tracing the
  simple-structure equation, sign convention, identification handling,
  recovery metrics, and citations.
- Orientation-invariant latent-map recovery metrics covering item parameters,
  person traits, person and item interaction positions, and distance weights.
- Scheduled, manual-dispatch, and release-tag statistical-study workflows that
  execute exhaustive ignored Rust studies in exact-name-validated shards while
  pull-request CI retains bounded CPU/GPU sentinels.
- A source-backed finite-Monte-Carlo convergence floor
  (`p0 - 2 * sqrt(p0 * (1 - p0) / R)`) for the 500-replication higher-order
  DINA recovery study.

#### Fail-closed Vuong selection summary

- A bounded public `compare_nonnested_models` orchestration API that preserves Rust-computed casewise likelihood-ratio mean, variance scale, corrected selection statistic, and two-sided probability together with explicit model-relation metadata when the normal-selection kernel is applicable.
- Auditable `ModelRelation`, `ComparisonStatus`, and immutable `ModelComparisonResult` contracts.
- Relation-appropriate routing for nested, boundary-nested, overlapping, strictly non-nested, and unknown candidate pairs.

#### Rubric blueprint compiler

- Versioned rubric and rubric-level schemas with explicit construct, observable evidence, task-family, response-format, locale, and prohibited-pattern contracts.
- Deterministic bounded compilation across task family, difficulty band, evidence mode, and replicate cells.
- Full SHA-256 rubric, blueprint, and generation-contract fingerprints plus authoritative 128-bit public blueprint and contract handles; 64-bit convenience display identifiers remain explicitly non-authoritative, and 64-bit digest slices also seed deterministic generation.
- A prompt-injection boundary and strict generated-item JSON Schema 2020-12 contract without adding a hosted-model SDK or network dependency.
- Immutable rubric and blueprint provenance constants in generated-item schemas, preventing wrong-blueprint replay from passing structural validation.
- Response-format-specific, closed, bounded answer-key contracts and ordered score-level schemas that require every rubric score exactly once.
- Explicit text and collection bounds for model-generated content and provenance fields.
- A deterministic standard-library changelog-fragment renderer; files in `docs/changelog.d` are authoritative `Unreleased` release notes and are validated as part of the repository test suite.
- Evidence-Centered Design documentation and a production roadmap from provider adapters through Rust-backed calibration and governed item-bank lifecycle.

#### Rust bifactor scoreability indices

- Rust-native continuous-indicator bifactor scoreability diagnostics: ECV-SS,
  ECV-SG, ECV-GS, item ECV, strict-pattern PUC, omega total, omega
  hierarchical, and construct replicability H.
- An explicitly named logistic latent-response conversion for fitted orthogonal
  bifactor slopes. Its omega values are documented as continuous
  latent-response coefficients, not categorical observed-score reliability.
- A modular PyO3 `_bifactor_core` surface and immutable typed Python API:
  `bifactor_scoreability`, `bifactor_scoreability_from_logit_slopes`, and
  `BifactorScoreabilityResult`. Python validates shapes and marshals results;
  all scoreability arithmetic remains in Rust.
- Fail-closed structural validation requiring every item to load on the
  declared general factor, uniquenesses in `[0, 1]`, and the standardized
  identity `sum(lambda^2) + uniqueness = 1` within `1e-8`.
- Formula-oracle, Rust/Python parity, structural, numerical-stability,
  logistic-conversion, and package-export tests plus buyer-facing
  interpretation boundaries.

#### Governed rubric item generation

- Bounded source-document packets with exact-content SHA-256 provenance and redacted audit metadata.
- Content-addressed generation requests that bind one rubric contract, blueprint, seed, and evidence-mode-valid source packet.
- A runtime-checkable provider protocol and deterministic offline fixture provider without hosted SDK, credential, or network dependencies.
- Strict provider-JSON decoding that rejects duplicate keys, non-finite numbers, oversized output, excessive nesting depth, missing fields, and unknown fields.
- Immutable rubric and blueprint replay protection across ids, 128-bit audit handles, full fingerprints, and governed rubric versions.
- Exact ordered rubric-score coverage, response-format-specific typed answer keys, option/key consistency, source-id resolution, and verbatim evidence-span validation.
- Explicit pairwise left/right/tie semantics with null-only tie preferences.
- Deterministic request, candidate, and execution fingerprints plus provider-failure redaction that omits raw source and generated text.
- Public generation, candidate, answer-key, attribution, and execution APIs with complete package exports.
Structural validation remains separate from semantic review, psychometric calibration, DIF, local-dependence, exposure, drift, and governed item-bank acceptance.

#### Adaptive factor rotation and criterion selection

- Rust-native adaptive exploratory factor rotation with a broad criterion
  registry, orthogonal and oblique gradient-projection optimization,
  deterministic multi-start search, coarse CPU multithreading, and explicit
  convergence/basin diagnostics.
- Criterion-neutral empirical selection using stability, simple structure,
  degeneracy, target recovery, bootstrap Tucker congruence, Pareto evidence,
  and declared decision policies. Objective values are never compared directly
  across criterion families or described as a proven global optimum.
- Modular PyO3 `_rotation_core` bindings and package-root Python APIs for
  criterion discovery, analytic value/gradient evaluation, multi-start
  rotation, and typed immutable solutions.
- Symmetric positive-definite Cholesky log-determinant/inverse handling for the
  Bentler criterion, including pivot-provoking and near-singular regression
  oracles.
- GPArotation-compatible complete/partial target semantics using binary
  zero-or-one masks and the loss `sum(w * residual^2)`. Continuous weights are
  available only through the separately named `lp_wls` kernel.
- An explicit scope boundary: Promax, Cubimax, iterative Lp/FSS orchestration,
  cluster/EIV/echelon procedures, user-defined compiled criteria, and a
  parity-verified wgpu batch optimizer are not part of this release slice.

#### Hourly pull-request governance

- A read-only hourly GitHub Actions loop that runs the existing pull-request queue governance evidence builder, publishes its JSON and accessible HTML audit artifacts, and retains native branch-protection and auto-merge gates instead of bypassing review or required checks.

### Changed

#### Rust-only literature true-parameter recovery gate

- The duplicate NumPy-only recovery experiment is removed; the Rust core is
  the single evidence path for literature recovery gates.
- The historical `cdm::tests::mc_ho_recovery_500` study is removed at the
  source level. Its generating design, fixed seeds, and RMSE, bias, and
  agreement thresholds are preserved verbatim by the reviewed
  `higher_order_dina_recovery_respects_monte_carlo_tolerance` integration
  study, which gates convergence on the documented two-standard-error binomial
  floor instead of an exact finite-sample proportion.

#### CI queue and review-governance hardening

- Pull-request CI runs now share a PR-number-scoped concurrency group, so a newer head cancels superseded queued or running CI evidence instead of consuming capacity for an obsolete commit.
- Push CI remains isolated by branch or ref and does not collide with pull-request validation.
- Draft pull requests no longer consume automatic CodeRabbit reviews, and automatic incremental review-on-every-push is disabled; maintainers request a final review only after a stable head is ready.
- The hourly read-only PR-governance workflow verifies its repository contract with the Python standard library, fails closed when no matching test is discovered, and no longer assumes that `pytest` is preinstalled on a fresh scheduled runner.
No test, security, packaging, coverage, or merge requirement is weakened by these operational changes.

### Fixed

#### Rust-only literature true-parameter recovery gate

- Ignored-test shard discovery rejects stale skip declarations, duplicate
  skips, ambiguous final-component exclusions, and silently empty shards.
- Explicit-GPU parity evidence fails closed when the Vulkan adapter is
  unavailable instead of silently skipping.

### Security

#### Fail-closed Vuong selection summary

- Omitted relation metadata defaults to `unknown`.
- Nested, boundary-nested, and unknown relations are routed before the non-nested normal-selection kernel is invoked, so a rejected or exact-zero non-applicable statistic cannot mask the required likelihood-ratio or relation-resolution procedure.
- The API does not report a winning model until Vuong's formal first-stage distinguishability evidence is available from a common compiled score/information contract.
- Numerical variance checks are not mislabeled as the formal weighted-chi-square distinguishability test.
- Casewise inputs are bounded and normalized to finite floats before FFI; booleans, opaque values, non-finite values, conversion overflow, malformed labels, invalid parameter counts, and compiled-kernel rejections fail closed without leaking low-level exception text or reproducing statistical arithmetic in Python.

## [0.1.2] - 2026-07-31

### Added
- Full paired-comparison / rating / inter-rater stack on main (PR #374 integrating the #290–#328 seonghobae chain tip): Thurstone Case V, Bradley–Terry MM, LSR/I-LSR, Rank Centrality, Plackett–Luce rankings and top-1, Kendall circular triads / *u*, Elo / Glicko / Glicko-2 / Stephenson / multiplayer Elo / FIDE, prediction metrics, BRATT ties model, Fleiss/Light kappa, ICC, Krippendorff α, Finn, Maxwell RE, Robinson *A*, mean pairwise Pearson/Spearman, Stuart–Maxwell / Bhapkar marginal homogeneity, rater bias, and Cohen kappa sample-size helpers — Rust core + PyO3/Python API + unit/paper tests.
- DeepWiki badge on the primary README docs surface (PR #373).

### Changed
- Stack features land without regressing the simple-structure MLS2PLM NLL path, coarse person-shard multithreading, or PRIMARY-only wgpu GPU init with soft f64 CPU fallback (preserved from v0.1.1 / PRs #371–#372).

## [0.1.1] - 2026-07-31

### Fixed
- Coarse fixed-shard Rust multithreading for the JML `neg_loglik_and_grad` hot path (`thread::scope` person shards, N≥256) with bit-identical reduction vs single-thread (PR #371).
- wgpu GPU init uses `Backends::PRIMARY` only (no GL/EGL), so sandboxes with broken `/dev/dri` soft-fail to the f64 CPU path instead of SIGSEGV (PR #371).
- Paper-grounded simple-structure MLS2PLM formula contract reaffirmed (Kang & Jeon 2025; Jeon et al. 2021); no formula drift.

### Changed
- Unit test forces multi-worker NLL shards and compares all gradient blocks to the single-thread reference.

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
