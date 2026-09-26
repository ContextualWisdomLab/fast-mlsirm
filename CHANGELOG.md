Warning: truncated output (original token count: 138420)
Total output lines: 6685

# Changelog

## Unreleased

### Added

- Opt-in `progress` callback on `fit_two_tier_grm` / `fit_bifactor_grm` exporting
  per-E-step marginal loglik / Δloglik (Bock & Aitkin, 1981, pp. 445, 447–448;
  #2021). Default remains silent.

### Changed

- Polytomous person fit now requires convergence by default, including for
  duck-typed fits with unknown convergence. `allow_unconverged=True` permits
  diagnostic use only for legacy or duck-typed fits and marks the result
  `valid_person_fit=False`, `diagnostic_only=True`; unconverged `PolyFipcFit`
  always raises. Provenance retains convergence and termination fields (or
  `"unknown"`). `PolytomousFit` retains its established standard-normal
  EAP grid even with non-default `prior_mean`/`prior_sd`, which affect only
  the `r0` correction. `PolyFipcFit` uses its fitted focal prior for both.

<!-- BEGIN AUTHORITATIVE CHANGELOG FRAGMENTS -->
### Added

#### Opt-in EM progress for long two-tier / bifactor GRM fits (#2021)

- Add optional `progress` callable on `fit_two_tier_grm` and `fit_bifactor_grm`
  (default `None`, silent — 0.11.4-compatible). Each E-step reports
  `EmIterationProgress(iteration, loglik, delta_loglik, start)` using the
  already-computed observed-data marginal log-likelihood (Bock & Aitkin,
  1981, *Psychometrika, 46*(4), pp. 445, 447–448). No extra quadrature.
- Rust companions `fit_*_with_progress` keep existing silent entry points
  unchanged; PyO3 detaches only when `progress is None`.

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

### Fixed

#### Red `python` gate from capability lanes another job owns

- The fail-closed outcome gate (`tests/conftest.py`, Issue #1732) escalated
  capability-gated skips in the ordinary `python` matrix. The first repair
  over-broadly allowed the whole high-q module and incorrectly described the
  Atheris harnesses as evidence for a separate Hypothesis module.
- The allowlist now names six exact high-q pytest nodes instead of a module
  glob. `gpu-smoke` installs a software Vulkan adapter, sets
  `STAGE5_HIGH_Q=1`, executes those six nodes plus the existing marginal GPU
  parity node, and fails when its JUnit evidence contains any skip.
- `tests/test_fuzz_properties.py` is no longer allowlisted. The `fuzz` job
  installs the `.[fuzz]` dependencies, executes that Hypothesis module
  directly, and rejects collection-time or runtime skips before running the
  separately owned Atheris harnesses. `hypothesis` (already in the `dev`
  extra) is also added to the hash-locked `requirements/ci.txt`, so the
  `python` matrix executes the module instead of collection-skipping it;
  without that the gate still failed on `collection-skip:
  tests/test_fuzz_properties.py` (#2075 run 106114739240).
- `test_allowlisted_capability_nodes_have_exact_ci_owners` prevents a future
  module glob, owner-name substitution, or allowlisted node without an
  executable CI command.

#### Red Semgrep gate on every PR

- The central `Semgrep (multi-language SAST)` gate reported three blocking
  WARNING findings on `main`, so it failed on every pull request regardless of
  its contents. The org ruleset gates on that workflow passing, so this blocked
  merges repository-wide. Reproduced locally with the same ruleset
  (`semgrep --config=p/default --severity=WARNING --severity=ERROR`), which
  returns the same three.
- `python/fast_mlsirm/dif.py` built its deprecated-alias docstrings by indexing
  `globals()` with loop variables drawn from a literal table three lines above.
  The rule cannot see that the keys are literals, and the indirection bought
  nothing: the loop now names the function objects directly, so a typo fails at
  import instead of at runtime, and the finding disappears with cleaner code.
- `tools/inventory_public_api.py` now enumerates repository-owned Python files
  and parses otherwise-unloaded public modules with `ast` instead of importing
  discovered module names. A regression fixture proves an import-time side
  effect in a discovered module is not executed, while constructor projections
  retain dataclass, enum, protocol, exception, and inherited signatures.
Neither change weakens the gate: both dynamic execution primitives are removed,
with no suppression or rule downgrade.
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
- Normalize model and stopping controls before caller response materialization across the CDM calibration, validation, model-selection, higher-order, and sequential entry points, preserving a consistent fail-closed control boundary.
- Preserve binary and ordered-category `NaN` missingness, Q-matrix validation, and Rust ownership of CDM likelihoods, marginal-ML EM, parameter estimation, classification, model-selection/validation statistics, higher-order structure, sequential-category arithmetic, and convergence.

#### IRT response, mask, and readiness-control admission integrity

- Reject complex, textual, object-backed, and arbitrary array-provider response evidence before float64 marshalling at the shared IRT response/readiness boundary.
- Preserve exact NumPy real-numeric arrays and ordinary built-in nested response containers, including supported concrete NumPy real scalar cells, `longlong`/`ulonglong` integer aliases, and NaN missingness.
- Reject callback-bearing mask evidence before Boolean coercion while preserving exact Boolean/real-numeric NumPy arrays and ordinary trusted numeric mask containers.
- Detect cyclic built-in response and mask containers with active-path identity tracking so shared acyclic rows remain valid while self/mutual cycles fail closed before NumPy materialization.
- Validate response rank, minimum persons/items, and a 20,000,000 logical-cell ceiling before contiguous float64 allocation, preventing large zero-stride or otherwise oversized exact arrays from forcing dense copies before rejection.
- Bound trusted built-in response/mask tree traversal before NumPy sequence materialization, charge logical cells hidden in exact NumPy row leaves, and reject zero-cell container fan-out that exceeds the structural-work envelope while preserving every valid 2-D matrix inside the 20,000,000-cell contract.
- Traverse built-in evidence one child at a time so peak traversal-stack memory is bounded by nesting depth rather than sibling fan-out, including malformed zero-cell fan-out rejected by the structural-work ceiling.
- Validate IRT family and category-count semantics before `fit_irt_experiment()` can materialize caller response evidence, and apply trusted response/mask admission before any production numerical fitter runs.
- Reject caller-defined integer subclasses and arbitrary integer-conversion providers at IRT experiment-readiness controls before caller callbacks can run, while preserving exact built-in and concrete NumPy integer scalar compatibility and existing readiness domains/errors.
- Keep production psychometric/statistical arithmetic Rust-owned; these changes are limited to Python validation, bounded materialization, and marshalling.

#### Bound CDM evidence before dense materialization

- Reject response and Q-matrix evidence above 20,000,000 logical cells during callback-free preflight, including oversized exact NumPy leaves nested in trusted built-in sequences, before NumPy materialization or `float64` allocation while preserving existing valid evidence and Rust-owned CDM arithmetic.
- Memoize trusted shared-sequence subtree sizes so repeated acyclic DAGs retain per-occurrence logical-cell accounting without exponential re-traversal, while true cycles still fail closed.
- Keep Boolean response round-trip validation compatible with the declared NumPy floor by reserving `equal_nan=True` for floating response arrays; non-floating admitted evidence uses ordinary exact equality.

#### Bound RSM responses before dense materialization

- Reject Rating Scale Model response evidence above 20,000,000 logical cells during the callback-free source preflight, including oversized exact NumPy arrays and exact NumPy rows nested in trusted built-in sequences, before NumPy stacking or contiguous `float64` allocation while preserving existing response semantics and Rust-owned Andrich arithmetic.

#### Bound polytomous prediction resources

- Reject public GRM/GPCM prediction grids above 20,000,000 dense probability cells before compiled-core discovery or output allocation.
- Apply the same 20,000,000-cell ceiling inside the Rust `polytomous_predictions` owner before item-parameter validation or `Vec::with_capacity`, so direct core/PyO3 callers cannot bypass the public resource envelope.
- Keep GRM/GPCM category-probability and expected-score arithmetic in the existing Rust implementation; the added checks govern request size and allocation only.

#### Polytomous prediction evidence admission

- Reject callback-bearing, complex, non-numeric, lossy, and non-finite GRM/GPCM prediction evidence before the raw prediction delegate or compiled-core discovery, including mixed built-in sequences whose integer identity would be lost by NumPy promotion.
- Preflight theta, slope, and category-parameter rank/resource contracts together and enforce the 20,000,000-cell joint output ceiling from trusted shape metadata before any NumPy materialization or float64 copy.
- Preserve trusted exact NumPy and built-in sequence inputs while keeping category-probability and expected-score arithmetic in the Rust prediction kernel; Python performs validation, bounded materialization, and marshalling only.

#### Polytomous prediction evidence rank admission

- Reject trusted-but-over-rank `theta`, slope, and category-parameter evidence before NumPy materialization or compiled-core discovery.
- Preserve exact 1-D theta/slope and 2-D category-parameter inputs, including exact NumPy row arrays nested in trusted built-in category matrices.
- Keep the existing prediction/evidence cell ceilings and Rust-owned GRM/GPCM probability and expected-score arithmetic unchanged.

#### Polytomous prediction category-domain admission

GRM/GPCM prediction admission now enforces the fitter-supported `2..=64` category domain at both the public Python boundary and direct Rust `polytomous_predictions()` boundary. Manually constructed `PolytomousFit` evidence above 64 categories fails before NumPy/native work, while direct native requests above `POLY_MAX_CAT` fail before prediction-grid allocation or item-parameter validation. Probability and expected-score arithmetic remain Rust-owned and unchanged.

#### Preserve polytomous prediction fit metadata

- Preserve all package-owned `PolytomousFit` convergence, trace, stopping, and threshold metadata when the public GRM/GPCM prediction admission boundary normalizes slope and category arrays before Rust dispatch.

#### Preserve strict GRM threshold order during initialization

- Preserve strictly decreasing finite GRM category thresholds for sparse or collapsed observed-category patterns by using the existing positive-pseudocount cumulative frequencies directly instead of independently clipping adjacent cumulative probabilities onto the same boundary. Returned GRM fits therefore remain inside the shared scoring and prediction parameter domain without changing Samejima category-probability arithmetic or GPCM behavior.

#### Polytomous raw prediction category-domain replay

- Replayed the fitter-supported `2..=64` category domain in the package-private Python prediction helper before resource calculation or compiled-core discovery, while preserving the valid 64-category boundary and keeping GRM/GPCM probability arithmetic Rust-owned.

#### Selection utility semantic-domain admission

- Replay the Rust-owned Brogden-Cronbach-Gleser/Naylor-Shine and Taylor-Russell input domains before compiled-core dispatch so trusted but invalid scalar controls fail at the Python validation boundary.
- Preserve callback-free exact Python/NumPy real-scalar admission while keeping selection-intensity, utility, bivariate-normal quadrature, success-ratio, and all result-affecting arithmetic Rust-owned.

#### Crossed estimator evidence admission

- Sealed the public crossed/multiple-membership person-effect estimator's response, fixed-item, slope, and person-offset evidence before NumPy array protocols can run. Exact NumPy numeric arrays and ordinary built-in list/tuple evidence containing package-trusted Python/NumPy numeric scalars remain supported, while arbitrary array providers, callback-bearing nested values, complex storage, and non-numeric storage fail closed before native-core discovery.
- Require exact integer and wider-than-binary64 real evidence to survive float64 normalization losslessly; values such as `2**53 + 1` and higher-precision `np.longdouble` values now fail closed instead of silently rounding before Rust dispatch, while exactly representable values remain supported.
- Bound crossed-response evidence to 20,000,000 logical cells before dense float64 materialization, charge nested exact NumPy rows by logical size, and bound malformed zero-cell container traversal without growing transient state with sibling width.
- Apply the same 20,000,000-cell pre-materialization envelope to item intercepts, optional item slopes, and optional person offsets, including exact broadcast vectors and nested exact NumPy vector leaves, so malformed fixed/person evidence cannot request dense float64 work before shape validation.
- Seal the public weighted-contextual-effect `worker_count` before comparison/conversion callbacks; caller-defined integer subclasses fail closed and supported concrete NumPy integer controls are normalized to package-owned built-in integers before Rust dispatch.
- Normalized admitted evidence to package-owned float64 arrays before delegating to the existing Rust-owned Fox–Glas/Browne MMMC MAP/Newton estimator; likelihood, updates, centering, GPU/CPU reductions, convergence, and recovery arithmetic are unchanged.

#### Conformance JSON decoder depth preflight

- Reject conformance-manifest JSON whose structural nesting exceeds `MAX_MANIFEST_NESTING` before invoking Python's recursive JSON decoder, while preserving the exact nesting boundary and ignoring bracket/brace characters inside quoted strings and escapes.
- Preserve the existing UTF-8/byte ceiling, duplicate-member and non-finite rejection, iterative post-parse nesting validation, canonical replay, and inventory-fingerprint checks.

#### Crossed continuous evidence admission

- Preserve Boolean response compatibility while rejecting Python and NumPy Boolean values in crossed/MMMC item intercepts, item slopes, and person offsets before native discovery, preventing silent `False`/`True` to `0.0`/`1.0` reinterpretation of continuous scientific evidence.

#### Contextual-effect Boolean admission

- Reject Python and NumPy Boolean identities in continuous contextual random-effect values before native discovery, while preserving the existing one-read mapping snapshot, supported real/integer effect values, and Rust-owned weighted-effect arithmetic.

#### Contextual-effect scalar admission

- Seal `weighted_contextual_effect()` continuous contextual-effect values before numeric conversion callbacks. Package-trusted Python/NumPy integer and floating scalars are normalized losslessly to inert binary64 values, while Boolean, complex, callback-bearing, non-finite, and lossy values fail closed before native discovery.

#### ATA target-curve evidence admission

- Reject callback-bearing, non-real, complex, and binary64-lossy target-theta or target-information evidence before NumPy materialization or item-information work, while preserving trusted NumPy/built-in numeric target curves and the historical single-point scalar target-information contract.
- Preserve exact real-numeric NumPy array rows nested inside inert built-in target trees without reopening array-provider callbacks; nested arrays are charged by logical size and replayed for lossless binary64 identity before materialization.
- Bound trusted ATA target evidence at 20,000,000 logical cells, built-in target nesting at 64 levels, and the dense target-point × item information matrix at 20,000,000 cells before per-cell conversion, NumPy materialization, psychometric scoring, or dense allocation; built-in tree traversal now keeps transient state proportional to nesting depth and bounds malformed zero-cell fan-out.
- Avoid a second Python per-cell lossless replay when assembly passes its already-normalized exact float64 target grid through the public item-information-matrix boundary; shape, finiteness, resource, and independent public-input validation remain intact.

#### Seal dichotomous CAT administration evidence

- Reject callback-bearing top-level array providers, ndarray/container subclasses, and non-real storage for partial CAT administered-item and response evidence before NumPy materialization or Rust ability-estimation dispatch.
- Preserve exact NumPy and ordinary built-in list/tuple numeric evidence, including concrete NumPy scalar compatibility, while retaining lossless signed-64 item-index validation, item range/uniqueness rules, and the exact 0/1 response contract.
- Reject over-rank, length-mismatched, and structurally impossible partial administrations from inert container metadata before value-wise scans or dense `int64`/`float64` marshalling; a validated EAP/MLE administration cannot exceed the calibrated bank item count because administered identities must be unique.
- Apply the over-bank EAP/MLE administration bound before inspecting the response carrier, so an unsupported response provider cannot force dense validation of an already impossible administration.
- Preserve `ability_standard_error`'s historical set-valued mask semantics: duplicate-laden and multidimensional administered evidence is normalized losslessly and deduplicated with `np.unique`, so the uniqueness-specific raw-length/rank preflight is not applied to that surface.
- Bound `ability_standard_error` administered-mask evidence to 20,000,000 logical cells from inert exact-container metadata before signed-64 value scanning, dense conversion, or `np.unique`, without imposing EAP/MLE uniqueness or rank semantics on the set-valued mask.
- Keep CAT probability, likelihood, EAP/MLE posterior/scoring, Fisher-information selection, stopping, and uncertainty arithmetic Rust-owned; this change is Python validation, bounded materialization, and marshalling only.

#### Bind releases and package artifacts to reviewed source commits

- Require manual release publication to name the exact reviewed release source commit, prove that commit is on the current protected default-branch lineage and is the commit that introduced both the requested project version and its released CHANGELOG section relative to its first parent, validate release metadata from that commit, and create or resume the immutable version tag only when it targets that same commit. This prevents either a later default-branch commit or an unrelated same-version descendant from being silently included in an already-cut release.
- Carry that same canonical release commit into package publication, verify the immutable version tag still peels to it, and build every sdist and wheel from the explicit commit rather than independently resolving the tag. The package workflow is selected from the protected default branch, and the release workflow now passes its exact dispatch commit as `control_plane_commit`; publication fails closed if the default branch advances before the downstream workflow is dispatched, so a moving branch cannot silently substitute a different publication-control definition.
- Research basis: Ohm, Plate, Sykosch, and Meier (2020), *Backstabber's Knife Collection: A Review of Open Source Software Supply Chain Attacks*, DOI `10.1007/978-3-030-52683-2_2`, analyzes 174 malicious packages distributed through npm, PyPI, and RubyGems. The methodological implication for this release path is to minimize mutable supply-chain identities: artifact source and the workflow control plane are carried as explicit immutable commits and checked again at the boundary where they are consumed.

#### Callback-safe, bounded, and lossless inference evidence admission

- Seal Hessian/covariance matrix identity before NumPy materialization for second-order, covariance, and standard-error diagnostics. Exact real-numeric NumPy arrays and inert built-in square matrices remain supported; arbitrary array providers, subclasses, complex storage, and non-numeric storage fail before caller protocols or Rust dispatch.
- Validate and normalize `tol` and `rcond` as finite non-negative Rust `f64` controls before caller matrix work. Boolean, callback-bearing, non-finite, negative, and lossy controls fail closed.
- Preserve exact 0-D real-numeric NumPy arrays as inert scalar controls for `step`, `tol`, and `rcond` when their values are losslessly representable in Rust `f64` and satisfy the existing semantic domain. Boolean, complex, object/text, non-0-D arrays, ndarray subclasses, and lossy/non-finite controls remain fail-closed before caller data work or Rust discovery.
- Seal `observed_information(..., step=...)` as an exact supported Python/concrete NumPy real scalar before config normalization, parameter packing, objective/data work, or native discovery. Boolean, callback-bearing, non-finite, non-positive, and lossy step controls fail closed; accepted values cross the finite-difference path as one package-owned built-in float.
- Apply a 20,000,000-logical-cell ceiling to trusted square Hessian/covariance evidence before dense `float64` materialization. Exact NumPy matrices are charged from inert shape metadata, and built-in square dimensions are bounded before row replay, preventing zero-allocation broadcast views or oversized built-in matrices from triggering unbounded dense allocation.
- Require every admitted matrix entry to preserve its numeric identity through Rust `f64` normalization. Built-in and concrete NumPy integers or wider floating values that would silently round during binary64 conversion fail before native inference work; exactly representable values and the existing non-finite covariance-diagonal semantics remain supported.
- Preserve Rust ownership of finite-difference Hessian coefficients/assembly, positive-definiteness eigendiagnostics, inversion/pseudoinversion, and covariance-diagonal standard-error arithmetic.

#### Seal subscore scientific-evidence admission

- Reject caller-defined array, container, and numeric protocol providers for Haberman subscore response and item-to-subscale evidence before NumPy materialization or compiled-Rust discovery.
- Preserve exact NumPy numeric arrays and exact built-in list/tuple trees of trusted concrete Python/NumPy numeric scalars or exact real/complex numeric NumPy array leaves while retaining complex, shape, completeness, and subscale-domain validation.
- Bound response/group scientific evidence to 20,000,000 logical cells before dense NumPy materialization, charge exact NumPy leaves from inert size metadata, reject true built-in-container cycles, and cap structural traversal independently while preserving shared acyclic subtrees.
- Keep Cronbach alpha, observed and disattenuated correlations, Haberman PRMSE, augmented-score weights, and added-value decisions in the Rust numerical owner; this change is Python validation, bounded materialization, and marshalling only.

#### Bound CRM response traversal before NumPy materialization

- Continuous Response Model response admission now applies an independent structural-work ceiling while traversing exact built-in list/tuple evidence, so malformed zero-cell or deeply nested container fan-out cannot consume unbounded Python work before NumPy materialization.
- The structural ceiling is `2 ×` the existing 20,000,000 logical-cell envelope, which preserves every valid non-empty two-dimensional persons-by-items built-in matrix while bounding malformed container-only traversal.
- Existing callback-free cycle rejection, shared acyclic subtree handling, exact NumPy row compatibility, logical-cell accounting, NaN-only missingness, and Rust-owned Samejima CRM likelihood/EM/EAP arithmetic remain unchanged.

#### Preserve Rasch CML control and group identity at the Rust boundary

- Reject `fit_rasch_cml()` and `andersen_lr_test()` tolerance controls that cannot be represented exactly as Rust `f64`, including wider `numpy.longdouble` values and oversized integer values, before caller response/group materialization or compiled-core discovery.
- Preserve distinct finite non-negative integral Andersen group identities carried by wider concrete NumPy floating scalars instead of narrowing them through Python `float` before deterministic dense-ID construction.
- Preserve exact built-in and supported NumPy scalar controls and group labels while keeping conditional likelihood, information, optimization, Andersen LR, p-value, and all other production psychometric arithmetic in the Rust core.

#### Seal parallel-analysis scientific evidence admission

- Reject caller-defined array/container/numeric protocol providers before Horn/Glorfeld parallel-analysis evidence can trigger NumPy coercion or compiled-core discovery.
- Preserve exact real-numeric NumPy arrays and exact built-in list/tuple matrices containing package-trusted Python/NumPy scalar evidence, then marshal accepted observations to contiguous `float64` for the Rust numerical core.
- Preflight the known two-dimensional carrier structure without recursive traversal, so over-rank built-in trees fail with the package shape contract instead of exhausting Python recursion.
- Require finite integer and extended-precision floating observations to preserve numeric identity through the Rust `f64` boundary, including before mixed built-in evidence can trigger NumPy dtype promotion.
- Preserve complex/non-real diagnostics, callback-free integer controls, and the 128 MiB random-eigenvalue workspace ceiling without changing observed/random eigenspectrum, bias adjustment, centile benchmarking, or retained-factor arithmetic.

#### Preserve G-theory mastery-cut identity at the Rust boundary

- Reject finite integer and extended-precision mastery cuts when binary64 normalization would change the threshold used by Rust-owned Brennan–Kane `Phi(lambda)` calculations.
- Apply the same lossless cut contract to the direct `phi_lambda()` API and the provenance-safe G-theory pilot handoff, so provenance cannot advertise a threshold that numerical marshalling changes.
- Preserve exactly representable built-in and concrete NumPy integer/floating controls as package-owned built-in `float` values while keeping Boolean, numeric-subclass, protocol-provider, and non-finite controls fail-closed.
- Leave G-study ANOVA/EMS, D-study variance components, Brennan–Kane signal and `Phi(lambda)` arithmetic unchanged and Rust-owned.

#### DIF pilot invariant replay

- Replay `DifPilotDesign` reference/focal group, row-alignment, identifier, and schema invariants before group-array construction, observed-score DIF argument projection, and canonical serialization/fingerprinting so frozen-record rebinding cannot silently change the populations supplied to DIF analysis.
- Require the wrapped binary pilot design to be the exact package-owned `MirtPilotDesign` record and replay group assignments only from an exact inert tuple, preventing caller-defined record/container subclasses from executing field, length, or iteration callbacks at the handoff boundary.
Production Mantel-Haenszel, logistic DIF, SIBTEST, purification, effect-size, and significance arithmetic remains unchanged and Rust-owned.

#### PR queue bounded JSON import

- Remove the plain-`json.loads()` isolation fallback from PR queue snapshot capture so queue-governance evidence always uses the repository's bounded JSON parser or fails closed when that parser is unavailable.
- Distinguish a missing package-layout import from a `ModuleNotFoundError` raised inside the real bounded parser, preserving the sibling direct-script import path without masking broken parser dependencies.
Existing GitHub retry deadlines, capture budgets, malformed-list evidence rejection, duplicate/non-finite/depth JSON policy, and queue identity limits remain unchanged.

#### Interaction-map evidence admission

- Seal `axis_count` before caller matrix work and reject callback-bearing integer identities, booleans, nonpositive controls, and requests above the interaction-map coordinate envelope before scientific evidence is inspected.
- Admit only exact real-numeric NumPy arrays or exact built-in two-dimensional numeric sequences before NumPy materialization, reject complex/non-real storage and infinities, preserve `NaN` as missingness, and require wider integer/floating evidence to survive the Rust `f64` boundary without changing identity.
- Bound public and native interaction-map logical cells and coordinate requests at 20,000,000 cells, and bound the Rust symmetric-eigendecomposition workspace at 128 MiB before dense Gram/eigenvector allocation.
Gabriel symmetric factorization, singular values, coordinates, reconstruction, unexplained residuals, cross-term arithmetic, and other production numerical behavior remain Rust-owned and unchanged.

#### Interaction-map empty complete-case result

- Normalize an empty complete-case interaction rectangle to empty respondent and item index sets so the Rust result remains shape-consistent with zero-length person/item coordinates and zero-by-zero reconstruction, unexplained-residual, and cross-share arrays at the Python boundary.
- Preserve the requested bounded `axis_shares` length for an empty map without inventing a maximal-complete-submatrix selection rule or retaining one non-empty axis after the other has collapsed.
Gabriel factorization, singular values, coordinate arithmetic, reconstruction, unexplained residuals, and cross-term calculations for non-empty complete-case rectangles remain unchanged and Rust-owned.

#### Interaction-map expected-evidence finiteness

- Preserve `NaN` exclusively as an observed-response missingness marker while rejecting observed infinity and rejecting both `NaN` and infinity in fitted model expectations before complete-case filtering or Rust factorization.
- Replay the same missingness/finiteness contract in the Rust core so direct PyO3/core callers cannot silently turn infinite observed evidence or invalid model expectations into missing cells that change the analyzed interaction rectangle.
Gabriel factorization, singular values, coordinates, reconstruction, unexplained residuals, distance, cross-term arithmetic, and observed-response `NaN` missingness remain unchanged and Rust-owned.

#### Judge projection mapping admission

- LLM-as-a-Judge construct projection now requires exact built-in criterion-score and criterion-category dictionaries before any mapping iteration or lookup. Caller-defined mapping protocols therefore cannot synthesize or replace criterion evidence during the IRT handoff, while exact `dict` inputs preserve the existing explicit item-order and category semantics.

#### Judge panel category-generation semantics

- LLM-as-a-Judge construct projection now requires one category-generation mode per persons-by-items panel. A panel may use explicit zero-based criterion categories for every row or derive categories from criterion scores for every row, but it cannot mix those response-generation semantics across respondents. The first row that changes mode is rejected before projection, while all-explicit and all-score-derived panels preserve the existing authoritative criterion order and Rust-owned GRM/GPCM numerical path.

#### Preserve RSM tolerance identity through Rust f64

- Reject Rating Scale Model `tol` controls whose exact Python or concrete NumPy integer/floating identity would change when marshalled to the Rust `f64` boundary.
- Preserve exactly representable built-in and NumPy controls, including supported `np.longdouble` values, while keeping callback-bearing scalar subclasses fail-closed before response materialization or native discovery.
- Keep RSM likelihood, marginal-ML EM/ECM updates, shared-threshold estimation, latent-trait integration, convergence, and scoring arithmetic unchanged and Rust-owned.

#### Seal Warm WLE controls and scientific evidence before caller protocols

Dichotomous and polytomous Warm WLE entry points now validate and normalize `theta_bound`, `tol`, category-count, and model-family controls before caller array materialization or compiled-core discovery. Callback-bearing scalar providers fail closed without executing their conversion protocols, accepted Python/NumPy numeric controls must preserve their exact value through the Rust `f64`/native-integer boundary, and supported NumPy string model identities are normalized to package-owned strings.
Explicit `observed` masks now use a callback-free Boolean admission boundary. Exact Boolean NumPy arrays and exact built-in list/tuple masks containing concrete Python/NumPy Boolean values are normalized to contiguous package-owned Boolean arrays; generic array providers, container/array subclasses, object/text/complex storage, non-Boolean cells, and shape mismatches fail closed before truth coercion or Rust discovery.
Item parameters and response evidence now pass an inert numeric-storage preflight before NumPy materialization as well. Exact real-numeric NumPy arrays and exact built-in list/tuple trees containing package-trusted Python/NumPy numeric scalars or exact numeric NumPy leaves remain supported; arbitrary array/numeric providers, subclasses, object/text storage, and cyclic container evidence fail closed before caller conversion protocols. Concrete complex evidence retains the existing field-specific `must be real-valued` diagnostic.
WLE scientific evidence is additionally bounded before dense marshalling. Exact NumPy arrays and logical occurrences of trusted sequence evidence may contain at most 20,000,000 cells, and built-in container traversal has a separate 40,000,000-node budget so zero-cell/deep fan-out cannot evade the logical-cell envelope. Exact NumPy leaves are charged by logical `size` before float64 allocation, shared acyclic sequence subtrees retain occurrence semantics without exponential re-traversal, and cycles remain fail-closed. Warm correction, information, root-search, standard-error, GRM, and GPCM numerical arithmetic remain Rust-owned.

#### Callback-safe and bounded Oakes uncertainty evidence admission

- Validate the Oakes finite-difference step `h` as a finite positive, losslessly representable Rust `f64` control before any caller response, factor, or mask evidence is inspected.
- Seal Oakes response and item-to-dimension evidence before NumPy materialization. Exact real-numeric NumPy arrays and inert built-in list/tuple evidence with concrete Python/NumPy numeric scalars remain supported; arbitrary array/numeric providers, subclasses, object/text storage, and concrete complex evidence fail closed with stable field-specific diagnostics.
- Seal optional observation masks before Boolean coercion so caller truth-value protocols cannot alter which response cells enter uncertainty estimation. Built-in mask cells use the same NumPy typecode-derived exact scalar universe as the response/factor admission path, preserving concrete integer aliases such as `longlong`/`ulonglong` without reopening subclass or protocol callbacks.
- Bound Oakes response evidence to 20,000,000 logical cells and built-in container traversal to 40,000,000 structural nodes before dense `float64` marshalling, while preserving `NaN`/`-1` response missingness and signed-64 factor narrowing contracts.
- Preserve Rust ownership of the Oakes information identity, finite-difference cross term, covariance, and standard-error arithmetic; Python changes are validation, bounded marshalling, and regression evidence only.

#### Benjamini-Hochberg evidence admission

- Validate the public Benjamini-Hochberg FDR control and p-value evidence before compiled-core discovery, reject callback-bearing, infinite, out-of-range, or lossy inputs without caller coercion, preserve the Rust-owned `NaN` missing-p-value contract, normalize accepted evidence losslessly through the Rust `f64` boundary, and keep historical package exports bound to the same hardened Rust-backed callable.
- Bound admitted BH evidence before value-wise or dense NumPy work: exact NumPy arrays and nested exact NumPy leaves are charged against a 20,000,000 logical-cell ceiling, while built-in list/tuple traversal has a separate 40,000,000-node budget so empty/deep fan-out cannot evade the logical envelope.

#### Fail-closed compiled Rust loader handling

- Normalize a discoverable but unloadable compiled Rust core to a package-owned runtime error while preserving the original loader exception as its cause.
- Reject non-string and `str`-subclass backend/device control values before caller-defined conversion or normalization callbacks can execute, while preserving case/whitespace normalization for exact built-in strings.

#### Runtime contract buyer-facing ownership

- Locked the Claude runtime-contract TOML block to package metadata and
  Rust-required `auto` ownership, and removed the stale buyer-facing claim that
  `auto` selected NumPy when the compiled core is missing. README, `FitConfig`
  comments, commercial Operational Notes, the buyer demo storyboard, sales
  `--check-import` help, PRD, TRD, and ADR-0002 now tell purchasers to install
  the Rust extension for production fitting. Explicit parity/reference work uses
  `fast-mlsirm fit --reference` at the CLI and the `fast_mlsirm.fit_reference`
  API in Python; direct production `fast_mlsirm.fit(...)` does not accept NumPy
  as a production backend. Release acceptance now rejects a NumPy outcome on
  `fit --backend auto`. The auto fail-closed error names the Python reference
  API without reflecting local paths or ABI details.

#### Harden configuration integer trust boundaries

- Reject caller-defined integer subclasses and arbitrary `__index__` providers before public simulation and fit configuration validation can dispatch caller-controlled coercion.
- Preserve exact built-in integers and genuine NumPy integer scalars while validating simulation size, optimizer-work, quadrature, latent-integration, seed, and verbosity controls through built-in integer values.
- Store those trusted integers back on the frozen configs so later size products and `seed + restart` cannot wrap narrow NumPy scalars.
- Normalize `dimensionality_diagnostics` `k_folds`, `seed`, and `latent_dims` to built-in integers before the candidate-by-fold budget product or `seed + fold_idx` can wrap a narrow NumPy scalar.
- Normalize `fit_diagnostics` `parameter_count` and `m2_q_*` to built-in integers before AIC/BIC arithmetic or `int(q_*)` can dispatch caller `__index__` hooks.
- Run the same simulation and fit validators at construction so memory-safety bounds cannot be bypassed by skipping an explicit `validate()` call.

### Security

#### Reject population-label int64 narrowing

- Reject unsigned values above the signed 64-bit boundary and floating-point
  values that would be saturated by NumPy during population-label compaction.
- Preserve the largest exact signed `int64` label while keeping group and
  cluster identifiers compact before Rust-owned allocation.

#### Seal rotation candidate-container admission

- Reject caller-defined rotation candidate-container subclasses before package-triggered iteration or Rust selector discovery, while preserving exact built-in list/tuple candidate sets and the existing exact-string criterion, uniqueness, policy, mode, and Rust-owned selection semantics.

## [0.9.0] - 2026-08-24


### Added

#### Cross-engine conformance inventory contract

- Add a provider-neutral, source-free `ConformanceInventory` contract for independent numerical conformance coverage. The first slice records public estimands, parameterization and identification scope, isolated engine/version/license identity, versioned parameter-mapping and fixture/environment fingerprints, and explicit passed/failed/indeterminate/not-executed states without adding external engines as runtime, build, package, or release dependencies. This is Python validation/provenance schema work only; production psychometric and statistical arithmetic remains Rust-owned.
- Accept both full Git SHA-1 and SHA-256 commit identities so protected-main and harness provenance remains valid across repository hash-format migrations.
- Require at least one executed evidence row before a capability can claim
  `covered` or `partially_covered` status.
- Revalidate exact package-owned engine, evidence, capability, and inventory records before manifest or fingerprint replay so post-construction field rebinding cannot bypass semantic-control, fingerprint, or collection admission; hostile enum controls and container subclasses fail closed before their callbacks execute.

#### Cross-engine conformance provenance

- Add optional run-level conformance provenance for the isolated harness
  commit, environment, RNG seeds, parameter-mapping schema, tolerance
  rationale, output fingerprints, and license classification without storing
  raw responses or adding an external-engine dependency.
- Revalidate exact run-provenance state before direct manifest replay so
  post-construction container rebinding fails closed before caller callbacks.

#### External validation profile contract

- Add a provider-neutral, source-free `ExternalValidationProfile` contract for preregistered external-validity and transportability evidence. The first slice keeps technical, construct, transportability, fairness, and decision-utility evidence distinct; preserves explicit failed/indeterminate/not-executed states; fingerprints normalized manifests; accepts provider-neutral dataset/site identities; and rejects evidence unavailable at the declared analysis cutoff. This is validation/provenance schema work only and does not move psychometric or statistical production arithmetic out of Rust.
- Reject caller-defined profile and evidence-record subclasses before reading their fields, keeping the immutable manifest boundary free of executable attribute callbacks.
- Reject overlapping development, internal-validation, and external-validation dataset identities so a transport claim cannot silently reuse a declared development cohort.
- Revalidate exact profile and evidence state before manifest or fingerprint replay so post-construction field rebinding cannot introduce hostile enum or container callbacks or make a manifest fingerprint disagree with its emitted payload.

#### Cross-engine runtime and redistribution provenance

- Bind cross-engine conformance runs to an explicit container-image or environment-lock identity, operating system, architecture, model-configuration digest, convergence-controls digest, and redistribution status.
- Include the new source-free runtime identities in deterministic manifests and inventory fingerprints while preserving exact-record replay validation against post-construction mutation.
- Keep external-engine evidence isolated from production numerical ownership; no psychometric or statistical arithmetic moves out of Rust.

#### Strict conformance manifest replay

- Added `ConformanceInventory.from_manifest()` and `from_json()` to rehydrate persisted cross-engine conformance evidence through exact package-owned validation.
- Persisted manifests now fail closed on unknown or missing nested keys, caller-defined mapping/list/text subtypes, duplicate JSON object keys, non-finite JSON constants, oversized JSON payloads, fingerprint tampering, and non-canonical normalized content.
- Replay remains provenance and serialization only; production psychometric and statistical arithmetic remains Rust-first.

#### Accessible cross-engine conformance evidence

- Add a deterministic standalone HTML and canonical JSON renderer for strict `ConformanceInventory` manifests, exposing capability coverage, capability × engine execution evidence, immutable inventory/run provenance, limitations, and explicit no-evidence states with exact values in text.
- Add a deterministic provenance-bound long-form JSON table so buyers can download one flat row per capability × engine evidence record without spreadsheet formula execution risk; capabilities with no independent engine remain explicit `not_executed` rows instead of disappearing or turning green.
- Escape manifest text, emit semantic table captions/headers and a restrictive no-script CSP, and state explicitly that numerical conformance is not construct validity, fairness, or high-stakes approval.
- Delegate all ingestion to strict manifest replay and keep the renderer reporting-only; no likelihood, discrepancy, RMSE/MAE, uncertainty, alignment, scoring, or other production psychometric/statistical arithmetic moves out of Rust.

#### Bind Figma buyer evidence to an authoritative ADR

- Record the buyer-review Figma File ID, packet-validation boundary, and
  downstream Code Connect ownership in ADR-0016 and the governance index.

#### Reproducible PyPI release publishing

- Added release-tag-bound sdist and wheel publication with a project-version provenance check, pinned Maturin and PyPA publisher revisions, and persisted checkout credentials disabled.
- The canonical release-tag workflow now explicitly dispatches package publication from the immutable tag, avoiding reliance on release events created with `GITHUB_TOKEN`, which do not recursively start ordinary event-triggered workflows.
- Isolated GitHub release-asset mutation from PyPI credentials, removed the unpinned runtime Twine installation path, and kept duplicate GitHub release assets and PyPI filenames fail-closed rather than silently replacing an immutable release artifact.
- PyPI publication now depends directly on the verified build artifacts rather than successful GitHub asset attachment, so a failed PyPI publication can be retried even when immutable release assets already exist and correctly reject replacement.

#### Crossed multiple-membership person effects

- Added a Rust-owned MAP estimator of crossed / weighted multiple-membership person effects `u_h` (Fox & Glas, 2001; Browne, Goldstein, & Rasbash, 2001). Persons may belong to several groups at once; one-hot nesting remains the singleton special case of the same sparse design.
- Added a CPU-multithreaded Bernoulli score/information reduction and an optional wgpu GPU kernel for that hot loop, with f64 CPU fallback when no adapter is present. Sparse Newton accumulation stays on CPU. This slice does not estimate OLS or AR longitudinal states.
- Added `fast_mlsirm.multilevel.estimate_crossed_person_effects` and `CrossedPersonEffectResult` as marshal-only Python access, plus a true-parameter RMSE recovery test against simulated crossed membership weights.
- Enforced the binary-response contract before native discovery and again inside the Rust estimator: finite non-negative observed cells must be exactly `0` or `1`; negative and non-finite cells retain the established missing-data semantics.

#### Govern structural-model pair decisions

- Add a governed structural-model selection gate that keeps factor retention separate from structure choice, requires explicit parameter-space relation evidence, refuses pairwise selection before the relation-appropriate LR/bootstrap/Vuong procedure, and gates any winner on recovery and intended-score interpretation evidence. The new Python surface performs validation and policy orchestration only; numerical comparison and psychometric arithmetic remain Rust-owned.

#### Add buyer-facing item-bank lifecycle reports

- Added deterministic JSON and standalone accessible HTML reporting for complete governed item-bank lifecycle lineages, including current state, rubric/blueprint provenance, approved-use scope, evidence-class inventory, transition timeline, and explicit missing-evidence limitations.
- Cross-version comparability is reported only as supported when governed linking evidence is present; the report never infers comparability from a nominal score range or active lifecycle state.
- Reporting remains provenance-only: calibration, fit, DIF, information, linking, exposure, drift, and uncertainty arithmetic are referenced by exact evidence identity and are not recomputed in Python.

### Changed

#### Govern non-psychometric item-bank suspension concerns

- Governed item-bank suspension and reactivation can now bind exact non-psychometric concern evidence for evidence/content validity and security/privacy findings, alongside existing DIF, drift, exposure, and linking evidence, without fabricating psychometric drift evidence.
- Suspended records bind the exact newly asserted concern classes into their content-addressed identity, and reactivation requires fresh evidence for those same classes so unrelated evidence cannot clear a quarantine.
- Reactivation rejects a historical approval or concern fingerprint even when it is presented under a replacement evidence identifier; every required reactivation artifact must bind new evidence content.

#### Production backend boundary

- Restrict production `FitConfig` and CLI backend selection to Rust (`rust` or
  fail-closed `auto`). Move the NumPy parity fit behind the explicit
  `fast_mlsirm.fit_reference` API and `fit --reference` mode, preserving
  testable parity without allowing an implicit production owner switch.
- Record the resolved Rust backend for the plain unidimensional MMLE fast path
  so CLI JSON and saved fit summaries report the execution owner rather than
  the unresolved `auto` selector.

#### Harden remaining equating controls before native discovery

- Validate circle-arc method/point/scalar controls, nominal-weights score ceilings and synthetic-population weight, and the composite-linking exponent before compiled-core discovery.
- Reject caller-defined scalar/container subclasses and arbitrary conversion providers without executing their conversion, comparison, representation, hashing, or iteration callbacks.
- Preserve exact built-in and genuine NumPy scalar compatibility while keeping circle-arc geometry, nominal-weights moments, composite-linking weight arithmetic, and all result-affecting equating mathematics in Rust.

#### Dedicated GRM recovery evidence retention

- Kept the 500-replication multidimensional Graded Response Model recovery
  study out of pull-request CI and out of the generic 1,800-second ignored-shard
  budget, then published its printed bias, RMSE, convergence, and theta
  correlation lines as a 90-day Actions artifact.
- Withheld checkout credentials from every Statistical Studies job so
  repository-controlled `cargo test` cannot reuse the Actions token.

#### Pin Rust 1.97.1 across verification

- Pin local Rust builds, Python/Rust package verification, ordinary Rust tests, GPU smoke, packaging, and scheduled statistical studies to exact Rust 1.97.1 instead of a floating stable channel.
- Track the root `rust-toolchain.toml` through Dependabot so future stable compiler updates arrive as reviewable pull requests with exact-head scientific, package, GPU, security, and recovery evidence.
- Preserve the existing public crate compatibility boundary by not adding or raising `package.rust-version`; this is a repository build-baseline change, not a new downstream MSRV claim.

### Fixed

#### Harden bounded subprocess cleanup

- Keep governance and procurement subprocess capture bounded across stdout, stderr, execution time, decoding, and JSON parsing. POSIX cleanup now avoids re-signalling an already reaped process group, successful capture closes parent-side pipe descriptors deterministically, and timeout/overflow paths retain fail-closed evidence without weakening repository gates.

#### Keep judge runtime validation active under Python optimization

- Replace production judge and calibration invariants that relied on removable `assert` statements with explicit package-owned `ValueError` or `RuntimeError` failures, and verify that invalid response-schema admission remains fail-closed under `python -O`.

#### Harden S-X² scalar control admission

- Reject caller-defined integer and floating subclasses at the public S-X² control boundary before numeric conversion or compiled-core dispatch, while preserving exact built-in and concrete NumPy scalar compatibility and leaving all S-X²/G², quadrature, and BH/FDR arithmetic Rust-owned.
- Reject built-in or concrete NumPy integer-valued real controls when float64 normalization would change the integer identity, so `min_expected`, `fdr_q`, and `min_effect` cannot be silently rounded before domain validation.

#### Item-bank transition replay callback safety

- Lifecycle transition replay now validates the exact creation-time record and evidence-reference instance state before invoking canonical serialization or fingerprint verification.
- Frozen lifecycle records mutated through Python object internals cannot shadow `_content_dict()` or evidence `to_dict()` callbacks to execute caller code while acquiring transition authority.
- This changes provenance/integrity validation only; calibration, fit, DIF, item-information, linking, exposure, drift, uncertainty, and other production psychometric arithmetic remain Rust-owned and unchanged.

#### Validate G-theory controls and score evidence before Rust discovery

- G-theory D-study sizes and `Phi(lambda)` scalar controls now fail closed before caller-owned score-array materialization and before compiled Rust capability discovery when invalid, while preserving the existing callback-free Python/NumPy scalar contract.
- D-study control containers now admit exact built-in list/tuple values and exact NumPy signed/unsigned integer arrays of the documented rank before iteration or pair unpacking, so caller-defined sequence, ndarray-subclass, and pair callbacks cannot run while `n_i_prime` / `n_prime` semantics are being established; concrete Python/NumPy integer entries remain supported.
- `gtheory_pi()`, `gtheory_pio()`, and `phi_lambda()` now reject callback-bearing array providers, non-real storage, and complex score evidence before NumPy real narrowing or Rust discovery; ordinary exact NumPy real arrays and built-in list/tuple score trees containing concrete Python/NumPy real scalars remain supported.
- G-study ANOVA/EMS, variance-component, D-study, and `Phi(lambda)` arithmetic remain unchanged and Rust-owned.

#### Harden Rudner/Lee cut-score control admission

- Validate and materialize Rudner and Lee cut-score scalars before compiled Rust capability discovery, rejecting booleans, caller-defined scalar subclasses, protocol coercion providers, malformed containers, non-finite values, and conversion overflow without invoking caller conversion hooks while preserving exact built-in and concrete NumPy real scalar compatibility. Both public paths now use one canonical package-owned normalizer; cut ordering/domain checks and all classification arithmetic remain Rust-owned.

#### Seal enterprise request record admission

- Enterprise issue scoring-request provenance now rejects caller-defined issue, stakeholder-perspective, and candidate-intervention record subclasses before reading their fingerprints or fields, preventing caller callbacks from executing during canonical record admission while preserving exact package record behavior.

#### Seal enterprise observation admission

- Enterprise issue observation admission now rejects caller-defined scoring-request, evidence-reference, and status-string subclasses before reading provenance or performing enum lookup, preventing caller callbacks during semantic validation while preserving exact package records and serialized status strings.

#### Seal enterprise explicit-value integer admission

- Reject caller-defined integer subclasses for enterprise explicit-value source offsets and deterministic parser record limits before comparison or coercion callbacks can execute, while preserving exact built-in integer domains and stable validation errors.

#### Seal scoring engine-authorization record admission

- Reject caller-defined assessment, scoring-request, and engine-descriptor subclasses before authorization policy or provenance fields are read, preserving exact package records, stable validation errors, and existing engine-policy semantics.

#### Seal assessment aggregate record admission

- Assessment assembly now rejects `ConstructSpec`, `RubricSpecification`, and scoring-policy subclasses before reading package-owned provenance or construct-scope fields, preventing caller-defined attribute/fingerprint callbacks from executing during aggregate contract admission while preserving exact package records and existing cross-reference semantics.

#### Bifactor scoreability control trust boundary

- Hardened both public bifactor scoreability entry points so `general_factor` and `zero_tolerance` are validated and normalized before loading, uniqueness, or logit-slope materialization and before compiled-core discovery.
- Reject booleans, caller-defined numeric subclasses, and arbitrary conversion-protocol objects without executing their callbacks, while preserving concrete Python/NumPy scalar compatibility and Rust ownership of index/domain validation and all scoreability arithmetic.

#### Scoring shared enum callback safety

- Shared scoring enum admission now preserves exact enum members and accepts only exact built-in strings for serialized enum values before invoking Enum lookup.
- Caller-defined string subclasses and arbitrary non-text objects fail closed with the existing package-owned assessment error before hostile hash or equality callbacks can run.
- Added public EngineDescriptor regressions proving callback-free rejection while preserving built-in string and exact enum-member compatibility; no scoring, calibration, likelihood, estimator, ranking, utility, or psychometric arithmetic changed.

#### Model-spec record admission

- Model resolution now admits only exact package-owned exploratory and confirmatory model records before reading their fields, so caller-defined model-spec subclasses cannot execute attribute callbacks during validation. Exact built-in/concrete NumPy factor counts and exact package model records retain their existing behavior; multidimensional exploratory estimation remains separately governed by #633.

#### Correct skewed-population Mokken study contract

- Keep the normal-trait Monte Carlo condition as the calibrated H/recovery
  contract.
- Standardize the positive-skew half-normal latent condition to the same
  location and scale as the normal condition before applying the shared 1.5
  theta scale, so the study changes distribution shape without confounding
  skewness with the previous approximately 28% narrower latent spread.
- Require both moment-matched latent conditions to retain the calibrated
  Loevinger H band, while keeping AISP full-recovery acceptance calibrated on
  the normal condition rather than treating the user-selected `c = 0.3`
  cutoff as distribution-invariant.
- Preserve the exact ignored-study execution and report failures normally.
- Declare that the workflow consumes no secrets and require reviewed
  `${{ secrets.NAME }}` environment injection for any future credentialed
  study.

#### Seal bounded JSON semantic-input callback boundaries

- Reject caller-defined byte/depth limit integers before comparison and caller-defined JSON text subclasses before encoding, while preserving exact built-in controls, bounded parsing semantics, and the existing descriptor/path/size/depth defenses used by repository release and governance automation.

#### Factor-retention callback safety

- Hardened governed factor-retention evidence admission so caller-defined integer and evidence-record subclasses are rejected before comparison or record-field callbacks can execute, while preserving built-in candidate counts and existing conservative retention semantics.

#### Harden multilevel text callback safety

- Require exact built-in strings for contextual schema versions, descriptive identifiers, and provenance fingerprints before comparison, normalization, regex, or encoding work, preventing caller-defined `str` subclasses from executing callbacks during multilevel and temporal contract admission.

#### Make repository test imports deterministic

- Pytest now exposes both the repository root and the Python source tree from
  committed configuration, so tests that materialize repository automation
  scripts do not require an operator-specific `PYTHONPATH=.` workaround.
- Agent guidance now derives its advertised Python support floor from the same
  `pyproject.toml` requirement guarded by repository tests, preventing stale
  lower-version setup instructions from diverging from package metadata.

#### Executed conformance provenance integrity

- Fail closed when a cross-engine conformance inventory contains executed `passed`, `failed`, or `indeterminate` evidence without exact run provenance.
- Require both raw-output and normalized-output SHA-256 identities for executed conformance runs while preserving optional output hashes for genuinely nonexecuted plans.
- Revalidate nested run provenance before applying the execution consistency gate so post-construction mutation cannot bypass package-owned admission.

#### Fail closed on missing release source identity

- The buyer-facing release evidence index now rejects timed-out, failed, unavailable, empty, malformed, or non-canonical Git `HEAD` identity instead of allowing an otherwise complete packet to report `status: "ok"` with unreconstructable source provenance.
- Valid repositories continue to record the exact full lowercase hexadecimal source commit without changing psychometric/statistical numerical ownership.

#### Require reconstructable buyer-packet source identity

- Buyer evidence packet generation now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated/malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving interoperability without changing psychometric/statistical numerical ownership.

#### Require reconstructable benchmark source identity

- Benchmark evidence generation now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated/malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving repository interoperability without changing psychometric/statistical numerical ownership.

#### Figma evidence source provenance

- Figma design-evidence manifests now fail closed when the repository source commit cannot be resolved to a canonical full lowercase SHA-1 or SHA-256 object identity, instead of emitting buyer-facing evidence with `source_commit: "unknown"` or an abbreviated/malformed revision.

#### Workflow registry transport failures

- The read-only workflow-registry audit now converts missing or inaccessible local GitHub CLI execution into a stable fail-closed `GitHubApiError`, so automation can emit bounded failure evidence instead of crashing with raw operating-system details.

#### Commercial release source identity

- Fail commercial release evidence generation closed when the source Git revision is unavailable or malformed, and require a canonical lowercase full SHA-1 or SHA-256 identity before provenance can be emitted.

#### Enterprise gate source-provenance hardening

- Require enterprise due-diligence manifests to bind `source_commit` to a canonical lowercase full SHA-1 or SHA-256 Git object identity instead of accepting abbreviated or arbitrary printable identifiers.
- Reject caller-defined string subclasses before text callbacks can execute at the source-provenance admission boundary, so a successful gate remains reconstructable from exact source identity.
- Restrict manifest output to a relative path inside the invocation directory and reject symlinked or tree-escaping destinations before writing.
- Write through a validated descriptor tree into a same-directory temporary file and atomically rename it into place on supported POSIX systems, so a failed write cannot truncate the previously accepted manifest.
- Preserve an existing manifest's access permissions across atomic replacement and use ordinary process file-creation permissions for a new manifest instead of forcing buyer-facing evidence to owner-only mode.

#### Enterprise gate semantic-control callback safety

- Reject caller-defined string subclasses for enterprise gate names and currency codes before normalization can invoke caller text callbacks.
- Reject caller-defined integer subclasses for procurement scenario amounts before comparison while preserving the positive-integer validation contract for exact built-in values.

#### Changelog fragment marker integrity

- Reject authoritative changelog fragments containing reserved managed-block marker literals before rendering or update, preventing nested markers from producing a changelog that fails its own next integrity check.

#### Scoring fingerprint text admission

- Require caller-supplied SHA-256 scoring provenance to be an exact built-in string before validation or retention, preventing valid-looking string subclasses from crossing the package trust boundary as canonical fingerprints.
- Apply the same exact built-in text boundary to structured scoring error code, path, and message fields.
- Reject caller-defined scalar subclasses in bounded scoring metadata before canonicalization or digesting.

#### Reject ambiguous duplicate JSON artifact members

- The shared bounded artifact JSON loader now rejects duplicate object member
  names at every nesting level instead of accepting last-value-wins semantics,
  while preserving its existing stable-file, UTF-8, byte, nesting, and parser
  controls.

#### Strict artifact JSON constants

- Reject `NaN`, `Infinity`, and `-Infinity` by default in the shared bounded artifact JSON loader so persisted package artifacts use interoperable JSON semantics; explicit caller `parse_constant` policies remain supported.

#### Require interoperable bounded JSON artifacts

- Reject duplicate object member names and non-standard non-finite numeric constants in the shared repository-automation bounded JSON reader, so file-backed and direct parsing use the same unambiguous RFC-compatible semantics while preserving existing size, depth, UTF-8, path-identity, and callback-safety controls.

#### Seal bounded subprocess command admission

- Reject caller-defined command-container and text-token subclasses before repository automation materializes or checks command arguments, preventing validation-time callback execution while preserving exact built-in list and tuple vectors.

#### Population-label narrowing safety

- Reject multigroup and multilevel population labels that cannot round-trip through signed 64-bit integer representation before compaction, preventing narrowing overflow from silently reordering the identified reference population while preserving valid sparse labels and the signed `int64` boundary.

#### BRATT control admission

- Validate and normalize Bradley-Terry-with-ties reference, iteration, and tolerance controls before comparison-data materialization or compiled-core discovery, rejecting callback-bearing scalar subclasses and protocol providers while preserving trusted built-in and NumPy scalar inputs.
- Keep BRATT probability, MM-update, reference-rescaling, convergence, and log-likelihood arithmetic unchanged in the Rust core.

#### RAG evidence limitation replay integrity

- Replay factory-derived RAG evidence limitation records before manifest or fingerprint projection so post-construction mutation fails closed before caller callbacks can execute.

#### Response-time calibration semantic control safety

- Reject caller-defined numeric and truth-value protocols before response-time calibration controls are normalized or dispatched to the Rust core.
- Require the joint speed-accuracy Gauss-Hermite node count to be an exact supported integer instead of silently narrowing floating-point values.
- Keep required positive-finite runtime validation active under optimized Python execution instead of relying on `assert` guards that disappear with `-O`.
- Preserve positive-finite stopping, variance-floor, and fixed-speed-scale contracts while keeping all response-time likelihood and estimation arithmetic Rust-owned.

#### Polytomous fit semantic control safety

- Reject caller-defined text, integer, real, and hashing protocols before GRM/GPCM calibration controls are normalized, response data are materialized, or the Rust core is discovered.
- Require calibration quadrature to use an exact supported integer node count rather than callback-capable membership or lossy coercion.
- Normalize both `NaN` and `-1` as missing polytomous responses before category validation, and report malformed response conversion through a stable package-owned numeric-input error.
- Preserve the category, iteration, and positive-finite stopping contracts while keeping the Bock-Aitkin EM/Newton estimator and all result-affecting psychometric arithmetic Rust-owned.

#### Keep essay-report pointer focus modality-safe

- Suppress pointer-acquired outlines on focusable essay-report table regions and canonical JSON blocks only when `:focus-visible` is false. Keyboard navigation retains the explicit high-contrast focus indicator, and regressions reject blanket `:focus { outline: none; }` suppression.

#### Reject overflowing polytomous DIF labels

- Polytomous DIF group and studied-item label/index vectors now verify signed-64-bit narrowing before compaction or Rust dispatch, preventing unsigned boundary values from wrapping negative and changing group/reference identity.
- Valid non-negative signed-64-bit and sparse/non-contiguous labels remain supported; GRM/GPCM DIF likelihood and statistical arithmetic remain Rust-owned and unchanged.

#### CAT administration data integrity

- Reject administered item indices that cannot be represented losslessly as signed 64-bit identities before range/mask handling, and reject complex-valued binary responses before any real-valued coercion can discard their imaginary component. Ordinary signed indices and real 0/1 responses retain the existing Rust-owned CAT likelihood, ability-estimation, and information paths.

#### Complex-valued polytomous response admission

- Reject complex-valued polytomous response matrices before any `float64` narrowing can discard imaginary components and turn a different observed category into a valid-looking real category.
- Preserve real integer categories plus `NaN` and `-1` missingness semantics across calibration, scoring, DIF, item/person fit, and other callers of the shared response-admission boundary without changing Rust-owned psychometric arithmetic.

#### CRM response data integrity

- Reject complex-valued continuous-response-model observations before NumPy can narrow them to `float64` and discard an imaginary component, and reject object-dtype response storage before caller-defined numeric conversion can run.
- Establish a callback-free response-evidence boundary before NumPy materialization: exact NumPy arrays and ordinary built-in list/tuple trees with package-trusted concrete Python/NumPy numeric scalars remain supported, while arbitrary array providers and caller-defined container/numeric subclasses fail closed before their protocols can execute. Exact numeric NumPy arrays nested as inert rows inside built-in containers remain compatible without admitting ndarray subclasses or object/text leaves.
- Preserve `NaN` as the CRM missing-cell marker while rejecting `+Infinity` and `-Infinity` before native discovery instead of silently reclassifying those invalid observed values as missing. Ordinary finite real-valued evidence retains the existing Rust-owned CRM fitting path.
- Bound CRM response evidence to 20,000,000 logical cells before sequence materialization or dense real-valued work. Exact broadcast arrays and exact NumPy row leaves nested in trusted built-in matrices are rejected from shape/size metadata before allocation; shared acyclic built-in subtrees retain logical-occurrence accounting without exponential re-traversal.

#### IRTree scientific-evidence admission

- Reject complex-valued IRTree response matrices, tree mappings, and node-dimension vectors before any `float64` narrowing can discard imaginary components and change observed categories, mapping branches, or factor assignments.
- Reject arbitrary NumPy array providers, callback-bearing container/scalar subclasses, and object/text storage before package-triggered `__array__` or numeric-conversion callbacks can synthesize or replace IRTree evidence.
- Preserve exact NumPy real-numeric arrays plus exact built-in list/tuple evidence containing package-trusted Python/NumPy real scalars, including ordinary `NaN` missingness, without changing IRTree mapping semantics or psychometric estimator arithmetic.

#### Complex-valued curvature admission

- Reject complex-valued Hessian and covariance matrices before any `float64` narrowing can discard imaginary components and alter second-order, covariance, or standard-error evidence.
- Keep eigendecomposition, inversion/pseudoinversion, and standard-error arithmetic in the Rust core while preserving existing real square-matrix contracts.

#### Oakes uncertainty input admission

- Reject complex-valued response matrices and factor assignments before any real/integer narrowing can discard imaginary components in the public Oakes standard-error wrapper.
- Preserve existing binary-response missingness and integer factor semantics while keeping Oakes information, finite-difference, inversion, and standard-error arithmetic in the Rust core.

#### Oakes factor-id signed-64 admission

- Reject Oakes `factor_id` values that cannot round-trip through signed 64-bit integer marshalling before dimension derivation or Rust uncertainty arithmetic, preventing unsigned overflow from silently changing item-to-dimension assignments.

#### WLE complex-evidence admission

- Reject complex-valued dichotomous and polytomous WLE responses and item parameters before real-valued marshalling or Rust scoring dispatch, preventing imaginary components from being silently discarded.

#### Seal LLTM data and control admission

- Reject complex-valued LLTM response matrices and explanatory-design weights before real-valued narrowing can discard their imaginary components.
- Validate Boolean, iteration, and tolerance controls before caller-owned data materialization or compiled-Rust capability discovery, while preserving trusted built-in and concrete NumPy scalar inputs and the Rust-owned LLTM estimator.

#### Nominal-response admission hardening

- Validate nominal category, quadrature, iteration, tolerance, Monte Carlo point, and RNG-seed controls before caller response materialization, accepting only package-trusted built-in or concrete NumPy scalar identities and passing normalized primitives to Rust.
- Reject complex response evidence before real-valued narrowing and reject infinite response values instead of silently reclassifying them as missing, while preserving ordinary real/integer categories plus documented NaN/negative missingness.
- Keep nominal probabilities, marginal likelihood, estimation, integration, convergence, identification, and EAP arithmetic unchanged in the Rust numerical core.

#### GPCM admission hardening

- Validate GPCM category, quadrature, iteration, tolerance, integration-point, and RNG-seed controls before caller response materialization, admitting only package-trusted built-in or concrete NumPy scalar identities and passing normalized primitives to Rust.
- Reject complex response evidence before real-valued narrowing and reject infinite response values instead of silently reclassifying them as missing, while preserving ordinary categories plus documented NaN/negative missingness.
- Keep GPCM probabilities, marginal likelihood, estimation, integration, reflection/identification, convergence, and EAP arithmetic unchanged in the Rust numerical core.

#### Mixture-response admission hardening

- Reject complex mixture-IRT response evidence before real-valued narrowing so caller data cannot silently project onto a different observed 0/1 pattern before Rust validation.
- Reject object-dtype response storage before per-element numeric coercion, including Python complex objects and caller-defined conversion callbacks, with the package-owned real-valued input error.
- Reject positive and negative infinity instead of treating them as undocumented missing responses, while preserving `NaN` as the documented MAR missingness representation.
- Keep mixture likelihood, posterior, EM updates, restart selection, canonical class ordering, convergence, and EAP arithmetic unchanged in the Rust numerical core.

#### KSIRT input admission

- Validate and normalize KSIRT kernel/grid controls before caller array materialization or compiled-core discovery, reject complex response or bandwidth evidence before real-valued `float64` marshalling, reject object/string-like storage before per-element numeric conversion can execute caller callbacks, and reject arbitrary array-protocol providers before NumPy materialization while preserving exact NumPy arrays and plain built-in numeric sequences. The Nadaraya-Watson/OCC estimator and all production psychometric/statistical arithmetic remain Rust-owned.

#### Mixed-format response admission

- Reject complex-valued mixed-format response evidence before real-valued marshalling so imaginary components cannot be silently discarded before categorical validation and Rust-owned calibration.

#### Subscore complex-evidence admission

- Reject complex-valued response and subscale-assignment evidence before real-valued marshalling so imaginary components cannot be silently discarded before Rust-owned Haberman subscore analysis.

#### DETECT evidence admission hardening

- Reject complex or non-real-numeric DETECT response storage before real-valued marshalling so observed binary evidence cannot be silently projected onto different data.
- Reject complex or non-real-numeric DETECT cluster storage before partition normalization so item-to-dimension labels cannot be silently projected onto a different real partition.
- Reject arbitrary response/cluster array-protocol providers before NumPy materialization, while preserving exact NumPy arrays and plain built-in sequences of trusted real scalar values.
- Reject a self-referential or otherwise cyclic list/tuple response or cluster (for example `a = []; a.append(a)`) before flattening instead of looping until the process is killed; cycle detection tracks only the active ancestor path, so legitimate repeated/shared acyclic rows remain accepted.
- Bound compressed shared-DAG list/tuple expansion and exact NumPy-array evidence before further package materialization, preventing hidden expansion or arrays above 20,000,000 logical cells while retaining ordinary shared-row compatibility.
- Preserve Rust ownership of conditional-covariance and DETECT index arithmetic; the Python change is limited to validation and marshalling.

#### Graded-response evidence admission hardening

- Normalize GRM integration, iteration, category, seed, and tolerance controls before caller response materialization, without invoking arbitrary scalar coercion callbacks.
- Reject complex, non-real-numeric, and infinite response storage before real-valued marshalling so observed graded-category evidence cannot be silently projected or reclassified as missing.
- Preserve the documented `NaN`/negative missingness convention, confirmatory loading validation, and Rust ownership of GRM likelihood, integration, parameter estimation, EAP, identification, and convergence arithmetic.

#### Linking evidence admission

- Reject complex-valued or non-real-numeric fixed-item and common-item linking evidence before lossy real marshalling, caller element conversion, or compiled Rust-core discovery; reject non-finite source-theta evidence before fixed-item Rust dispatch while preserving Rust-owned linking arithmetic.

#### Factor input admission hardening

- Reject complex and non-real-numeric factor-analysis, reliability, and Velicer MAP evidence before real-valued marshalling can alter caller data or execute object-element conversion.
- Normalize trusted `n_factors` and `max_m` integer controls before caller array materialization and Rust-core discovery while preserving concrete NumPy integer compatibility.

#### Parallel-analysis data admission

- Reject complex and non-real-numeric caller matrices before Horn/Glorfeld parallel-analysis input is narrowed to `float64`, preventing imaginary evidence from being silently discarded or object-element numeric callbacks from running during package-owned admission.
- Preserve existing real numeric input compatibility, integer-control validation, bounded random-eigenvalue workspace policy, and Rust ownership of eigenvalue, random-benchmark, centile, and retention arithmetic.

#### Validate Hofstee controls before score materialization

- Validate and order the four Hofstee percentage controls before caller-owned score arrays are materialized, so rejected semantic controls cannot trigger score-side array protocols before the package emits its stable validation error.
- Preserve the existing Rust-owned Hofstee ogive, intersection, fallback, and cut-score arithmetic.

#### CAT exposure item-evidence admission

- Reject complex-valued and non-real-numeric Sympson-Hetter and a-stratified item-parameter storage before lossy `float64` marshalling or compiled-core discovery, while preserving ordinary real item banks and Rust-owned CAT exposure algorithms.

#### Seal Chang-Ying KL evidence admission

- Reject complex or non-real-numeric KL item-parameter storage before any lossy `float64` narrowing or Rust-core discovery.
- Require `kl_select()` administration masks to use Boolean storage rather than truth-value coercion.
- Normalize `theta0`, `delta`, and `r` only from package-trusted built-in or concrete NumPy real scalar identities before caller array work.
- Preserve contiguous `float64`/Boolean native marshalling after admission while leaving Chang-Ying KL integration and selection arithmetic Rust-owned.

#### Delta-plot group evidence admission

- Reject non-real-numeric Delta-plot group storage before real-valued coercion, preventing textual reference/focal labels from being silently reinterpreted and object-dtype cells from executing caller numeric callbacks during Python-to-Rust admission.
- Preserve ordinary numeric and Boolean 0/1 group arrays while keeping Angoff Delta-plot psychometric arithmetic unchanged in the Rust core.

#### Owen CAT evidence admission

- Establish Owen posterior/CAT scalar, Boolean, item-array, and binary-response trust boundaries before compiled-core discovery or caller-controlled coercion. Caller-defined scalar/truth callbacks, complex/text/object item or response storage, and arbitrary array providers now fail closed while supported NumPy scalar/array evidence is normalized to inert built-in/contiguous representations. Owen posterior moments, b-matching, variance stopping, and all result-affecting psychometric arithmetic remain Rust-owned.

#### Seal EPV trust-boundary admission

- Reject caller-defined posterior scalar callbacks, lossy or non-numeric EPV item evidence, and non-Boolean administered masks before native dispatch while preserving ordinary NumPy inputs and Rust-owned predictive/variance/selection arithmetic.

#### Seal Sympson-Hetter scalar control admission

- Validate package-trusted `r_max` and `tol` scalar identity and semantic domains before caller item arrays or native discovery, preserving Rust-owned Sympson-Hetter calibration, simulation, update, and stopping arithmetic.
- Preserve the Rust finite `tol >= 0` contract directly in the canonical `exposure.sympson_hetter` boundary and remove the duplicate zero-tolerance marshalling/dispatch shim.

#### Seal SPRT evidence and control admission

- Validate package-trusted Wald SPRT scalar controls and reject coercive, textual, object, or complex item/response evidence before native dispatch, preserving Rust-owned boundaries, likelihood-ratio accumulation, first-crossing decisions, and trace arithmetic.

#### Seal CI-classification evidence and control admission

- Validate package-trusted confidence-interval classification controls and reject coercive, textual, object, or complex item/response evidence before native dispatch, preserving Rust-owned EAP, posterior-SE, interval, and strict first-crossing arithmetic.

#### Flexilevel evidence admission

- Validate Lord flexilevel item-count and platform-size controls before caller response materialization, and reject complex, textual, object-backed, lossy, or domain-invalid response/probability evidence before native-core discovery while preserving supported binary NumPy arrays, plain callback-safe 1-D/2-D list/tuple response array-likes, and finite odd-length probability vectors. Routing, red/blue self-scoring, forward recursion, score-lattice probabilities, mean, and variance remain Rust-owned.
- Preserve callback-safe list/tuple probability compatibility for package-trusted concrete NumPy real scalars as well as built-in real scalars.

#### Observed-score equating evidence admission

- Reject complex, object-backed, and textual score/frequency evidence before lossy `float64` marshalling or compiled-Rust discovery across equivalent-groups, NEAT, kernel, presmoothing, and SEE entry points.
- Preserve real Boolean/integer/unsigned/float evidence while keeping equating, smoothing, uncertainty, and population-linking arithmetic Rust-owned.

#### Fixed-form test assembly admission safety

- Harden fixed-form assembly so form length and content-constraint controls are normalized before caller item evidence, complex/object information cannot be projected through `float64`, content labels are admitted as text without caller stringification, and exclusion indices must fit signed 64-bit item identity without narrowing overflow before the Rust-owned greedy assembly runs.

#### Harden constrained-CAT evidence admission

- Validate CCAT ability, item, content-group, target, and administered-mask evidence before native dispatch; reject callback-bearing or lossy storage, require lossless non-negative integral `uintp` group marshalling, and leave constrained-CAT selection arithmetic Rust-owned.

#### Bound the judge's weighted-score boundary

- `ContextualOrchestratorJudge.judge()`'s plain scoring path (no `category_count`, the simplest public interface) trusted the model's own self-reported top-level `score` for the accept/reject decision instead of deriving it from `criterion_scores` and each `JudgeCriterion.weight`, unlike the three `category_count`-based paths, which already discard the self-reported score in favor of a mechanically recomputed weight-aware average. A model could report a high aggregate score while giving a low score on a heavily-weighted criterion and still be accepted. Made the plain path derive `score` the same way as the other three (issue #1238).
- Rejected a non-finite aggregate criterion weight before any contextual-orchestrator transport call. `JudgeCriterion` validates each weight as finite and positive, but two individually valid weights (for example `1e308` each) could still overflow their sum to infinity; a weighted score could then silently collapse to an incorrect finite value (for example `0.0`) instead of failing closed. All three weighted-score paths now share one bounded, finite denominator (issue #1235).

#### Response-time evidence admission

- Reject complex, object/text, callback-bearing, and arbitrary array-provider response-time evidence before real-valued marshalling or Rust-core discovery across standalone RT calibration, joint speed-accuracy calibration, and RT person-fit diagnostics, while preserving ordinary built-in real-numeric sequence and NumPy-array inputs.
- Replaced the recursive built-in-sequence walk with an explicit stack so a deeply nested response-time list/tuple (past Python's recursion limit) or a self-referential one (`a = []; a.append(a)`) rejects with a validation error instead of crashing the process with an uncaught `RecursionError` or looping forever.

#### Response-time person-fit control safety

- Validate `alpha_level` and `z_fast` with callback-free concrete real-scalar admission and the Rust-owned `(0, 1)` / finite non-negative domains before native-core discovery in response-time person-fit diagnostics.

#### Empirical Bayes DIF evidence admission

- Reject arbitrary array-protocol providers and callback-bearing sequence elements before Empirical Bayes Mantel-Haenszel DIF evidence is narrowed or dispatched, while preserving exact NumPy real-numeric arrays and ordinary built-in real-numeric list/tuple vectors.

#### Nonparametric person-fit response admission

- Reject arbitrary array-protocol providers and callback-bearing response cells before complete dichotomous person-fit evidence is materialized or dispatched, while preserving exact NumPy real-numeric arrays and ordinary built-in real-numeric list/tuple matrices.

#### DIMTEST evidence admission hardening

- Reject arbitrary response and AT1/AT2 array-protocol providers before NumPy materialization so caller callbacks cannot synthesize scientific evidence or subtest membership.
- Preserve exact NumPy real-numeric arrays and plain built-in sequences of trusted real scalars, plus existing complete dichotomous response and integer index semantics.
- Preserve Rust ownership of Stout DIMTEST conditional-variance, bias-correction, p-value, and retained-group arithmetic.

#### Seal paired rating-range evidence admission

- Reject callback-bearing or subclassed caller rating containers before NumPy conversion or Rust-core discovery, while preserving exact NumPy numeric arrays and the existing ordinal category/domain checks. Paired rating-range descriptive arithmetic remains Rust-owned.

#### Reliability evidence admission

- Reject callback-bearing, complex, or non-real-numeric caller evidence before Rust discovery in Guttman lambda, ten Berge mu, Cronbach alpha, and person-separation reliability entry points, while preserving ordinary NumPy arrays and trusted built-in sequence inputs.
- Reject over-nested or cyclic built-in sequence evidence at the public API's known 1-D/2-D rank boundary before NumPy materialization or native discovery, while preserving shared acyclic rows and trusted real-scalar sequence compatibility.
- Use one callback-free masked-array diagnostic across ICC, Guttman lambda, ten Berge mu, Cronbach alpha, person separation, and pairwise-rater reliability so masked evidence consistently tells callers to encode missingness with NaN before any native dispatch.
- Preserve historical built-in sequence compatibility when rows are exact real-numeric NumPy arrays, while retaining callback-free rejection of ndarray subclasses and non-real row storage before materialization.
- Preserve historical rater-sequence Boolean semantics without reopening caller protocols: pure Boolean built-in sequences keep the Boolean-specific diagnostic, while mixed Boolean+numeric built-in sequences retain NumPy's numeric promotion.
- Make reliability-adapter installation recover every primary sibling after an interrupted partial bind instead of treating a hardened ICC wrapper alone as proof that the whole public reliability surface was installed.
- Bound primary and rater reliability evidence to 20,000,000 logical cells before NumPy materialization or contiguous `float64` allocation, including exact broadcast views and exact NumPy leaves nested inside trusted built-in sequences.

#### Pairwise reliability evidence admission

- Validate the Pearson/Spearman pairwise-rater Fisher control and caller-owned ratings evidence before native discovery, rejecting callback-bearing or non-real evidence without changing Rust-owned correlation, ranking, Fisher-transform, or inference arithmetic.

#### ICC ratings evidence admission

- Preserve callback-free ICC semantic controls while also rejecting callback-bearing, complex, Boolean, or non-real ratings before native discovery; trusted numeric arrays and built-in numeric sequences still marshal to the unchanged Rust ICC implementation.
- Preserve the established Boolean-rating diagnostic for trusted built-in/NumPy-Boolean sequences, including mixed Boolean-plus-numeric sequences whose Boolean identity NumPy would otherwise erase by numeric promotion, and preserve actionable `NaN` missingness guidance for NumPy `MaskedArray` subclasses without reopening caller-defined array or scalar callbacks.

#### Remaining reliability rater-evidence admission

- Validate Krippendorff alpha, Finn reliability, Maxwell RE, and Robinson A semantic controls and rater evidence through callback-free package admission before Rust discovery, while preserving trusted numeric sequence compatibility and the existing Rust-owned agreement/reliability arithmetic.

#### Answer-copying evidence admission

- Reject callback-bearing NumPy array providers, ndarray/container subclasses, and caller-defined numeric subclasses before answer-copying evidence is materialized for Wollack omega, K-index/K1/K2/S1/S2, or GBT.
- Preserve exact NumPy numeric arrays and exact built-in list/tuple evidence containing package-trusted Python/NumPy real scalars, while keeping existing complex, dimensional, finite, index, binary, probability, and relation validation contracts.
- Keep all result-affecting answer-copying statistics and tail/regression arithmetic in the Rust numerical core; this change only hardens Python validation and marshalling.

#### Bound G-theory score evidence before dense materialization

- `gtheory_pi()` and `phi_lambda()` now reject score evidence outside the documented two-dimensional persons-by-items shape before dense NumPy materialization; `gtheory_pio()` applies the same fail-first contract to its three-dimensional persons-by-items-by-occasions shape.
- G-theory score evidence now has an explicit 20,000,000-cell logical-resource ceiling that applies to exact NumPy views and trusted built-in sequence trees before a contiguous `float64` copy is allocated.
- Built-in score-tree preflight now advances one child at a time, so transient traversal state is bounded by nesting depth instead of eagerly scheduling every sibling before the logical-cell ceiling can fire.
- Existing exact NumPy arrays, ordinary built-in list/tuple score trees, exact NumPy-array rows, callback-free cycle rejection, and Rust-owned G-study/D-study/`Phi(lambda)` arithmetic remain unchanged.

#### Bound G-theory D-study result-row requests

- `gtheory_pi()`, `gtheory_pio()`, and `phi_lambda()` now reject D-study request vectors above 10,000 rows before score materialization or compiled-core discovery.
- D-study result-row count is bounded independently from the existing 1,000,000 per-prime magnitude ceiling, so small valid prime values cannot be repeated to request an unbounded native result table.
- Exact built-in list/tuple controls, trusted Python/NumPy integer entries, the existing per-prime size bound, and all Rust-owned G-study/D-study/`Phi(lambda)` arithmetic remain unchanged.

#### G-theory NumPy D-study control compatibility

- Preserve exact NumPy signed/unsigned integer arrays for one-facet and two-facet D-study size controls, and preserve exact built-in `range` values on the one-facet `Sequence[int]` surface, while continuing to reject ndarray subclasses, arbitrary array providers, callback-bearing sequence subclasses, Boolean/float/object/text control arrays, malformed rank/shape, non-positive values, and existing resource-limit violations before Rust dispatch.
- Normalize accepted NumPy control arrays and built-in range controls to package-owned built-in integer payloads; G-study, D-study, and `Phi(lambda)` arithmetic remain unchanged and Rust-owned.

#### Rater reliability installer recovery

- Recover interrupted Krippendorff/Finn/Maxwell/Robinson reliability-adapter installation by requiring the complete package-owned rater wrapper set before idempotent short-circuiting, while preserving callback-free evidence admission and Rust-owned reliability arithmetic.

#### Close CI contract drift on the toolchain pin and metadata scalar admission

- Pin the `grm-recovery` scheduled statistical-study job's `dtolnay/rust-toolchain` step to exact Rust `1.97.1`, closing a gap where it silently floated to the default stable channel while every sibling verification lane stayed pinned.
- Align `test_metadata_normalizes_string_subclasses_without_callbacks` (formerly `test_metadata_rejects_string_subclasses_before_callbacks`) with the metadata scalar admission boundary's actual, intentional behavior: caller-defined `str` subclasses are safely normalized through the inert `str.__str__` descriptor (matching the established `int`/`float` subclass handling in the same function) without invoking any subclass-defined method, rather than being rejected outright.

#### Fail closed on unsafe multilevel contextual effects

- Multilevel contextual-effect evaluation now fails closed when any referenced context random-effect value is NaN or infinite and when finite inputs overflow the weighted sum, preventing non-finite predictor results from escaping the Rust boundary while leaving unreferenced table capacity outside sparse validation work.
- Python context-effect marshalling snapshots each required mapping value once without caller-defined membership probes and normalizes hostile lookup callbacks to non-reflective package errors before native dispatch.

#### Seal governed RAG request replay

- Reject caller-defined `ScoringRequest` subclasses at governed RAG perturbation and facets-calibration replay boundaries before any request field can execute caller code. Exact factory-sealed requests retain the existing provenance validation, while invalid subclasses now fail through stable non-reflective package errors.

#### Harden model-comparison casewise numeric trust boundary

- Harden public non-nested model-comparison casewise value admission so arbitrary float-protocol objects and caller-defined numeric subclasses fail closed without executing conversion callbacks, while preserving exact Python and supported NumPy real scalars; Vuong statistics remain Rust-owned.

#### Multilevel M2 moment and covariance ownership

- Move multigroup and multilevel M2 population-moment integration into the
  Rust/PyO3 numerical boundary, including the shared cluster-intercept
  reduction.
- Move the finite-cluster moment-covariance construction into Rust while
  preserving compact-label validation, finite-cluster correction, and the
  existing M2/RMSEA2 estimand.
- Keep the NumPy implementations available only as explicit parity references;
  public M2 paths fail closed when the required native entry point is absent.

#### Multilevel M2 Rust projection

- Multilevel M2 now routes both fitted-model and cluster-robust independence projections through the compiled Rust core, failing closed when that projection entrypoint is unavailable.

#### Structured M2 Rust ownership

- Route public single-population `m2()` calls that include estimated population
  moments, anchored items, or a fixed spatial coefficient through the Rust/PyO3
  M2 kernel. Missing structured native capability now fails closed instead of
  entering the NumPy reference implementation.
- Preserve the existing M2 estimand and degrees-of-freedom contract while
  moving finite-difference calibration and population nuisance columns into
  the Rust numerical owner.

#### Workflow-registry audit transport retry hardening

- Expanded the read-only Actions-registry audit transport's bounded retry classifier to cover transient HTTP 403, 404, 429, and all 5xx responses, while preserving fail-closed exhaustion and immediate failure for non-transient authentication errors such as HTTP 401.
- Added direct transport regression coverage so incident audits do not misclassify one transient GitHub control-plane response as a completed inventory failure.

#### Harden RAG metadata callback safety

- Validate caller-provided RAG metadata keys exactly once before reading any values, then freeze only the captured allowlisted values. Hostile membership, key/value, duplicate-key, and key-reiteration callbacks now fail through non-reflective package errors without granting new metadata authority.

#### Exposure-control scalar callback safety

- Validate CAT/exposure integer controls from exact built-in Python and genuine NumPy scalar types before caller-dispatchable coercion or Rust-core discovery, preserving integral built-in/NumPy floating controls, package-owned bounds/errors, and Rust-owned exposure, routing, scoring, posterior, recovery, and simulation arithmetic.

#### Harden scoring-policy integer callback boundaries

- Reject caller-defined integer coercion at scoring-policy positive-integer boundaries before any `__index__` callback can run, while preserving exact built-in and genuine NumPy integer scalar compatibility and existing bounded `AssessmentSpecError`…38420 tokens truncated…ERIC ED069624)**
  (`fast_mlsirm.livingston_k2`, `fast_mlsirm.livingston_correlation`; in
  Rust `mlsirm_core::classification::{livingston_k2,
  livingston_correlation}`): the classical-test-theory analogues of
  reliability and correlation with moments taken about a criterion
  (cut) score instead of the mean, `D^2(X) = var + (mean - cut)^2`,
  `k^2 = (rho^2 var + (mean-cut)^2) / D^2(X)`, and
  `k(X,Y) = D(X,Y)/sqrt(D^2(X) D^2(Y))`, with Spearman-Brown test-length
  projections applied to `k^2` itself. The conversion form is an
  algebraic reconstruction from the source's Table 1 expectation
  definitions; `k^2` is NaN only in the exact degenerate case (scores all
  exactly equal to the cut, detected element-wise, or `var == 0 &&
  mean == cut`), and returns the formula limit 1 when the squared
  criterion offset overflows f64 with finite variance (the correlation
  rejects that overflow with an error); fractional Spearman-Brown lengths are a
  disclosed continuous extrapolation. Exact-fraction anchors (k2 = 5/6,
  SB(2) = 10/11, sign-flip k = 5/7 with norm rho = -1, asymmetric-offset
  k = 22/(7 sqrt(10))), equality-iff-mean=cut and zero-variance property
  pins, error contracts, and a 500-rep Monte Carlo recovery check
  (`#[ignore]`).
- **Brennan-Kane index of dependability Phi(lambda) for mastery tests
  (Kane & Brennan, 1977, ACT Technical Bulletin No. 28, ERIC ED185076,
  eq. 33)** (`fast_mlsirm.phi_lambda`; in Rust
  `mlsirm_core::gtheory::phi_lambda`): the criterion-referenced
  dependability coefficient theta(d) = Phi(lambda) for a one-facet random
  `p x i` design at a cutting score `lambda`, built on the module's
  `gtheory_pi` ANOVA. The `(Xbar - lambda)^2` signal is estimated with a
  derived unbiased plug-in that subtracts `varhat(Xbar)` computed from the
  RAW (unclamped) variance components, while `sigma^2(Delta')` and the
  `sigma^2(p)` numerator keep the module's clamped-component policy; the
  signal is left unclamped, so estimates may fall below the lambda-free
  `dependability` (finite-sample behavior, documented). TB-28 defers
  estimation to Brennan & Kane (1977a, JEM), which was not read; the
  estimator is derived and adversarially verified independently.
- **Subkoviak single-administration coefficient of agreement (Subkoviak,
  1976, ERIC ED120229 / JEM 13(4))** (`fast_mlsirm.subkoviak_agreement`; in
  Rust `mlsirm_core::classification::subkoviak_agreement`): per-person and
  group coefficients of agreement, marginal chance agreement, and Cohen's
  kappa for mastery classifications under the simple binomial true-score
  model, with the regression estimate of the item-domain proportion
  (Eq. 16) and optional KR-21 reliability derived from the data with the
  population (ddof = 0) variance. Supports multi-category criteria
  (Eqs. 19-22); mastery convention is score `>= C`, verified against
  Table 1 of the read source (its Eq. 4 OCR prints `>`). The compound
  binomial refinement (Eqs. 12-14) and Lord's (1959) distribution-free
  estimate (Eq. 17) are excluded because they defer to sources not read.
  Exact-fraction oracle pins from the paper's Table 1 fixture; five
  executed mutation kills (category boundary, P(i) squaring, chance-term
  aggregation, KR-21 ddof, regression-weight swap).
- **Hanson-Brennan compound-binomial classification consistency and accuracy
  (Hanson, 1991, ACT Research Report 91-5)** (`fast_mlsirm.hanson_brennan`,
  `fast_mlsirm.hanson_brennan_from_params`; in Rust
  `mlsirm_core::classification::hanson_brennan` /
  `hanson_brennan_from_params`): single-administration decision consistency,
  accuracy, sensitivity, specificity, and Cohen's kappa for
  number-correct cut scores under a four-parameter beta true-score
  distribution with Lord's two-term approximation to the compound binomial
  conditional error model. The data path estimates Lord's k from the score
  mean/variance and reliability (Hanson, 1991, Eq. 6), recovers the first
  four true-score moments by the HB.tsm recursion (Eqs. 7-8), fits the
  four-parameter beta by the method of moments with a two-parameter
  failsafe (identical branch structure to `livingston_lewis`); the params
  path accepts explicit (l, u, alpha, beta, k). The conditional fail CDF
  uses a derived closed form
  `BinCdf(cut-1;K,p) - k p(1-p) [b(cut-1;K-2,p) - b(cut-2;K-2,p)]`,
  verified as an exact polynomial identity against Lord's term-by-term
  definition in the oracle. Pinned against an exact-Fraction stdlib oracle
  (params fixtures at 1e-12; a genuine negative-k 4P data fixture with both
  beta shapes < 1 at 1e-7); five mutation kills executed.
- **Two-stage adaptive testing (Betz & Weiss, 1973, Research Report 73-4;
  Betz & Weiss, 1974, Research Report 74-4)** (`fast_mlsirm.two_stage_route`,
  `fast_mlsirm.two_stage_score`; in Rust
  `mlsirm_core::exposure::two_stage_route` / `two_stage_score`, PyO3
  `py_two_stage_route` / `py_two_stage_score`): routing-test scoring via the
  truncated normal-ogive ability estimate theta-hat =
  Phi^-1(((x'/m) - c) / (1 - c)) / a-bar + b-bar (Equation 2; perfect scores
  truncate to m - 1/2, chance-or-below scores to c*m + 1/2), assignment of
  the measurement test whose mean difficulty is closest to the routing
  estimate (minimum absolute difference; ties break to the lowest index, a
  derived convention), and the item-count-weighted composite
  (m1*theta1 + m2*theta2)/(m1 + m2) (Equation 3). The scoring entry point
  re-derives the routing assignment and refuses a mismatched
  `administered` index so second-stage scores are never combined with the
  wrong measurement test's parameters. Anchored on the reconstructed
  Appendix B routing table of Research Report 74-4 and an exact-Fraction
  oracle through the p-computation; both subtests require m*(1-c) > 1 for
  distinct truncation endpoints.
- **Pyramidal adaptive testing (Larkin & Weiss, 1974, Research Report
  74-3)** (`fast_mlsirm.pyramidal_administer`; in Rust
  `mlsirm_core::exposure::pyramidal_administer`, PyO3
  `py_pyramidal_administer`): deterministic single-examinee replay of the
  classic up-one/down-one equal-offset pyramidal ("branched") design — items
  in a triangular structure ordered by difficulty (stage s holds s items,
  n(n+1)/2 total), a correct response routing to the harder stage-(s+1)
  neighbour and an incorrect response to the easier — with Larkin & Weiss's
  six scoring methods: number-correct, mean difficulty attempted, mean
  difficulty correct (NaN when indeterminate), final-item difficulty, the
  hypothetical (n+1)th-item "final difficulty score" (computed only when the
  caller supplies the next-stage difficulties; the paper's pool-specific
  column-mean construction is out of scope), and Hansen's all-item score as
  described by Larkin & Weiss (verified against the printed 15-stage 0–240
  range). Routing recurrence and all-item stage scores are DERIVED from the
  source prose (labelled in the module comment); exact-fraction oracle
  anchors, checked-arithmetic overflow guards, and a 500-rep Monte-Carlo
  structural invariant test (`#[ignore]`).
- **Weiss stradaptive (stratified-adaptive) test administration (Weiss, 1973,
  Research Report 73-3)** (`fast_mlsirm.stradaptive_administer`; in Rust
  `mlsirm_core::exposure::stradaptive_administer`, PyO3
  `py_stradaptive_administer`): deterministic single-examinee replay of
  Weiss's stratified-adaptive design — an item pool partitioned into S ≥ 2
  difficulty strata, up-one-stratum after a correct response and
  down-one-stratum after an incorrect response (edge-clamped, with a DERIVED
  fallback to the last administered stratum when the clamped target is
  exhausted), terminating on a ceiling stratum (≥ min_items administered and
  proportion correct ≤ chance), pool exhaustion, or max_items. Reports
  ceiling / basal / highest-non-chance strata, Weiss's ten ability scores
  m1–m10 (NaN when indeterminate; score 7 interpolates between adjacent
  stratum mean difficulties with side-dependent steps), and a consistency
  index (population variance of the score-9 stratum set; DERIVED — defined
  verbally in the report without a printed numeric anchor). The primary
  source was READ (ERIC ED084301); routing and scores are pinned by the
  report's William W. protocol (Fig. 2 / Appendix A) and its five printed
  score-7 cases plus synthetic below-chance anchors that discriminate the
  lower-step branch (the printed cases alone do not). Score 5's
  extrapolated-next-stratum variant for exhausted pools and free-response
  (chance = 0) termination are deliberately out of scope.
- **Lord self-scoring flexilevel testing (Lord, 1970, RB-70-43; Lord, 1971,
  RB-71-6)** (`fast_mlsirm.flexilevel_administer` /
  `fast_mlsirm.flexilevel_score_distribution`; in Rust
  `mlsirm_core::exposure::flexilevel_administer` /
  `flexilevel_score_distribution`, PyO3 `py_flexilevel_administer` /
  `py_flexilevel_score_distribution`): deterministic replay of Lord's
  branched-adaptive flexilevel design over a full 0/1 response matrix — N
  (odd) difficulty-sorted items, n = (N+1)/2 administered starting at the
  median (right → easiest harder, wrong → hardest easier), number-right
  self-scoring with +1/2 for a wrong last answer — plus the exact conditional
  score distribution f(x | θ) on the half-integer lattice {1/2, …, n} via
  Lord's forward recursion over p_v(i), taking caller-supplied per-item
  correct-response probabilities (ICC-agnostic). Both primary ETS Research
  Bulletins were READ (ERIC ED042813 / ED051286); the routing is pinned by
  Lord's RWWRWRRRWR worked example and the recursion is cross-checked exactly
  against exhaustive path enumeration. Lord's Eq. 3 efficiency ratio and
  Eq. 4 normal-ogive ICC are deliberately out of scope.
- **Breslow-Day odds-ratio homogeneity DIF test (Breslow & Day, 1980, Eq. 4.30)**
  (`fast_mlsirm.breslow_day_dif`; in Rust `mlsirm_core::dif::breslow_day_dif`,
  PyO3 `py_breslow_day_dif`): the classical NON-UNIFORM DIF companion to
  `mantel_haenszel_dif` — MH tests a common odds ratio against 1; this tests
  whether a common odds ratio is tenable at all across the matching-score
  strata. Per used stratum (all four margins positive) the fitted
  reference-correct count is the admissible root of the fitted-value quadratic
  `A·D/(B·C) = ψ̂` (cancellation-stable q-form roots; defensive
  both-roots admissibility check), the asymptotic variance is
  `1/(1/A + 1/B + 1/C + 1/D)` on the fitted cells, and
  `χ² = Σ (a − A)²/Var` is referred to χ²(K − 1). The plugged-in `ψ̂` is the
  crate's MH `alpha_mh`, the estimator the read source itself endorses
  (worked example: MH 5.158 → χ² 9.28 vs MLE 5.312 → 9.33). Degenerate MH
  odds ratio (`Σad = 0` or `Σbc = 0`), fewer than two usable strata, or an
  inadmissible fitted root yield NaN statistics; Benjamini-Hochberg flags are
  computed across items on the finite p-values. The Tarone (1985) correction
  and the Eq. 4.31 trend test are deliberately out of scope (sources not
  read/documented in the citation-governance header). 0/1 responses only.
- **Generalized Mantel-Haenszel nominal DIF (Zwick, Donoghue & Grima, Eq. 10)**
  (`fast_mlsirm.gmh_dif`; in Rust `mlsirm_core::dif::gmh_dif`, PyO3
  `py_gmh_dif`): unordered-category DIF screening — examinees matched on the
  full total score; within each usable stratum the reference group's
  category-count vector over `T − 1` categories is compared with its
  conditional expectation and covariance; the pooled quadratic form
  `d′S⁻¹d` is referred to χ²(`T_eff − 1`). Effective categories are counted
  in used strata only (categories seen solely in excluded strata do not
  inflate `df`); singular pooled covariance yields NaN (no silent rank
  reduction); category cap `T_eff ≤ 64`. For 0/1 items the statistic equals
  the `mantel_smd_dif` χ² (MH without continuity correction). Integer
  non-negative category codes only (reduced scope; no missing-data support).
- **Mantel polytomous DIF + standardized mean difference (Zwick, Donoghue & Grima)**
  (`fast_mlsirm.mantel_smd_dif`; in Rust `mlsirm_core::dif::mantel_smd_dif`, PyO3
  `py_mantel_smd_dif`): ordinal-item DIF screening — examinees matched on the
  full total score, per-stratum focal score sums compared with their
  conditional hypergeometric expectation/variance (Mantel χ², df = 1, Eqs. 8–9
  of ETS RR-93-14), plus the standardized mean difference effect size (Eq. 11,
  focal-weighted focal-minus-reference item mean difference; weights
  renormalized over usable strata — documented deviation matching the crate's
  standardized P-DIF convention). For 0/1 items the χ² reduces to the MH
  chi-square without continuity correction. Integer non-negative scores only
  (reduced scope; no missing-data support).
- **Empirical Bayes Mantel-Haenszel DIF (Zwick & Thayer)**
  (`fast_mlsirm.eb_mh_dif`; in Rust `mlsirm_core::dif::eb_mh_dif`, PyO3
  `py_eb_mh_dif`): shrinkage enhancement of MH D-DIF statistics — prior
  `N(μ, τ²)` estimated from the supplied item set (`μ` = mean, `τ²` =
  across-item variance minus mean squared SE, floored at 0), per-item
  posterior mean `W·MH + (1−W)·μ` and variance `W·SE²` with
  `W = τ²/(τ² + SE²)`, plus posterior probabilities of the five ETS DIF
  categories (`C−, B−, A, B+, C+`, normal areas delimited at ±1.5/±1).
  Formulas trace to the READ report Zwick & Thayer (2003, LSAC RR / ERIC
  ED481063, statistical-model section); the variance divisor (`n−1`) and
  the degenerate `τ² = 0` point-mass boundary conventions are documented
  implementation choices not printed in the source. Takes MH D-DIF/SE
  pairs (e.g. from `mantel_haenszel_dif`), so any MH pipeline output can
  be stabilized for small samples.
- **Angoff Delta plot DIF detection (deltaPlotR-faithful, response input)**
  (`fast_mlsirm.delta_plot`; in Rust `mlsirm_core::dif::delta_plot`, PyO3
  `py_delta_plot`): transformed item difficulties `4·qnorm(1−p)+13`, the
  R-compatible major axis with `max(b1, b2)` root selection (kept even
  under negative delta covariance, regression-tested), perpendicular
  distances, normal-approximation or fixed detection thresholds, extreme
  proportion handling (`constraint` clamp or `add` correction), and IPP1/
  IPP2/IPP3 iterative item purification with R's membership-row
  convergence semantics — ported from the CRAN deltaPlotR R package's
  `deltaPlot.R` and `adjustExtreme.R` (READ at commit e2aeeb6; Angoff &
  Ford 1973 and Magis & Facon 2012/2014 are cited only as implemented).
  Response-type input only (the R proportion/delta paths, printing, and
  plotting are out of scope); non-{0,1,NaN} responses are rejected rather
  than silently averaged, and returned item indices are 0-based.

- **Nonparametric person-fit statistics (PerFit-faithful, complete data)**
  (`fast_mlsirm.person_fit_np`; in Rust
  `mlsirm_core::personfit_np::person_fit_np`, PyO3 `py_person_fit_np`):
  seven dichotomous statistics — Guttman error count G, normed Guttman
  errors, the norm conformity index NCI, van der Flier's U3 and
  standardized ZU3, Sato's caution index C, and the modified caution
  index C* — ported from the CRAN PerFit R package's `G.R`, `Gnormed.R`,
  `NCI.R`, `U3.R`, `ZU3.R`, `C.Sato.R`, and `Cstar.R` (READ at commit
  c9df433; the originating papers are cited only as implemented).
  Complete 0/1 data only: PerFit's missing-value imputation and
  polytomous variants are out of scope, and any non-{0,1} entry is
  rejected. Perfect (all-0s/all-1s) rows are source-faithful: G, normed
  G, and NCI are 0 (the R source applies `1 - 2*Gnormed` before its
  NaN→0 replacement) while U3/ZU3/C/C* are NaN, and degenerate
  all-equal-difficulty data yields NaN rather than an error. Column
  ordering reproduces R `order(pi, decreasing = TRUE)` including its
  ascending-index tie-break, pinned by a dedicated tie fixture.

- **Hofstee compromise standard setting (psychometricsGP-faithful)**
  (`fast_mlsirm.hofstee`; in Rust
  `mlsirm_core::standard_setting::hofstee`, PyO3 `py_hofstee`): the
  Hofstee compromise cut score, a computational port of the
  psychometricsGP R package's `fn_plot_hofstee()` (`R/fn_plot_hofstee.R`,
  READ — the only inspectable implementation found; single-source port,
  stated openly; plotting excluded; Hofstee 1983 itself NOT READ, cited
  only as implemented). Intersects the piecewise-linear cumulative
  relative frequency ogive over integer score bins 0..=100 (right-closed
  bins `(s-1, s]`, divide-first `(count/n)*100` arithmetic preserved)
  with the descending diagonal `(min_cut, max_fail)` → `(max_cut,
  min_fail)`; when they do not cross, the R fallback pins the cut to
  `min_cut`/`max_cut` with a strict `<` fail count and two-decimal
  DIRECTED rounding (ceil up-branch / floor down-branch), `failed=True`.
  Reduced scope per adversarial spec review: collinear ogive-diagonal
  overlap and zero-length diagonals are rejected (`spatstat`
  `crossing.psp` degenerate semantics unverified against an R runtime).
- **K1/K2/S1/S2 answer-copying indices (CopyDetect-faithful)**
  (`fast_mlsirm.k_variants`; in Rust `mlsirm_core::security::k_variants`,
  PyO3 `py_k_variants`): the four regression-baseline copying indices,
  ported exactly from the CRAN CopyDetect package's internal `ks12()`
  (`R/similarity1.r`, READ), specialized to complete scored 0/1 data (no
  missing responses — the port rejects anything but exact 0/1). Number-
  incorrect subgroups EXCLUDE the source (the opposite of `k_index`'s base
  `k()` convention — a deliberate CopyDetect asymmetry, regression-anchored
  in tests). K1/K2 fit linear/quadratic least squares of subgroup incorrect-
  match rates and take binomial upper tails `P(Bin(ws, p) >= m)`; S1/S2 fit
  log-linear Poisson GLMs of (weighted) match counts and take bounded
  Poisson WINDOW probabilities (`P(m <= X <= ws)` / `P(mm <= X <= n_items)`,
  not plain upper tails — CopyDetect subtracts the tail beyond the cap),
  with S2 adding the `(1.5e)^(-6·prob)` weighted correct-match term
  and a RAW ceiling (`mm = ceil(sum) + m`, no epsilon — float noise at
  integer boundaries can bump `mm`, documented). Numerics: rank-checked
  modified Gram-Schmidt QR for the OLS fits (degenerate designs raise, no
  silent normal-equation blowup) and a guarded Newton Poisson GLM with
  step-halving, bounded eta, and a stable start (nonconvergence raises);
  `ks12()` itself SUPPRESSES R's non-integer-Poisson warning for the S2
  fit. Sotaridona & Meijer (2002, *JEM 39*(2)) and (2003, *JEM 40*(1)) NOT
  read — all four indices cited only as implemented by CopyDetect.
- **Generalized binomial test (GBT) tail kernel (aberrance-faithful)**
  (`fast_mlsirm.gbt`; in Rust `mlsirm_core::security::gbt`, PyO3 `py_gbt`):
  exact Poisson-binomial distribution of the copier-source match count via
  Bernoulli-convolution DP and the INCLUSIVE upper-tail p-value
  `P(M >= observed)`, ported exactly from the CRAN aberrance package's
  `compute_GBT` (`src/compute.cpp`, READ) and corroborated by CopyDetect's
  internal `GBT()` (`R/similarity1.r`, READ — same distribution, same
  inclusive tail). Per-item match-probability construction is the caller's
  job (aberrance directional and CopyDetect symmetric recipes both fit);
  missing data out of scope (the packages conflict). van der Linden &
  Sotaridona (2006) NOT read — cited only as implemented. Returns the full
  pmf plus the p-value; O(n^2) time / O(n) memory nonnegative f64 DP — no
  cancellation, but tiny extreme large-n masses may underflow.
- **K-index of matching incorrect answers (CopyDetect-faithful)**
  (`fast_mlsirm.k_index`; in Rust `mlsirm_core::security::k_index`, PyO3
  `py_k_index`): binomial upper-tail index of copier-source shared incorrect
  answers against a number-incorrect subgroup baseline, ported exactly from
  the CRAN CopyDetect package's internal `k()` (`R/similarity1.r`, READ;
  corroborated by `R/similarity2.r`), with the binomial tail summed in log
  space (no factorial overflow or extreme-p underflow). The subgroup
  includes the copier and,
  when scores match, the source (CopyDetect convention). Holland (1996,
  RR-96-07) and Sotaridona & Meijer (2002) NOT read — cited only as
  implemented; Sotaridona & Meijer (2001, ERIC ED467373) read for
  background. Validation rejects non-binary/complex/bool inputs and the
  degenerate all-correct source.
- **Omega answer-copying statistic (Wollack-style)**
  (`fast_mlsirm.wollack_omega`; in Rust `mlsirm_core::security::wollack_omega`,
  PyO3 `py_wollack_omega`): standardized index of answer similarity between a
  suspected copier and a source. `h` counts identical observed options,
  `p_i = P_i[source_i]` is the copier's model-implied probability of the
  source's observed option, `omega = (h - sum p_i)/sqrt(sum p_i (1 - p_i))`
  with a one-sided upper-tail normal p-value. Formula verified against two
  independently READ implementations: the CRAN CopyDetect R sources
  (`similarity1.r`/`similarity2.r`) and the aberrance package
  (`compute_OMG`); NOT read: Wollack (1997, *Applied Psychological
  Measurement, 21*(4), 307-320) itself (access blocked) — cited only as
  implemented by those sources. CopyDetect's printed docs flip the sign
  (`(E-h)/sqrt(V)`) but both source files use `(h-E)/sqrt(V)`; the source
  convention is implemented. Scope: omega only — no g2/GBT/K-index, no
  continuity correction, no missing responses; the caller supplies the
  copier's fitted option probabilities (e.g. from a nominal response model).
  Pinned against an independent Python oracle at 1e-12 (p-values 5e-7 via
  crate erfc); error paths, structural single-item-extension invariant, and
  a 500-rep Monte Carlo size/power check (`#[ignore]`); 3 executed mutation
  kills (V-vs-sqrt(V) scaling, copier-probability lookup, two-sided p).

- **DIMTEST test of essential unidimensionality (original Stout-style
  AT1/AT2 statistic)** (`fast_mlsirm.dimtest`; in Rust
  `mlsirm_core::detect::dimtest`, PyO3 `py_dimtest`): confirmatory
  hypothesis test with caller-supplied assessment subtests AT1/AT2 (equal
  length >= 4, disjoint) and the complementary partitioning subtest PT;
  examinees are grouped by raw PT total score (groups smaller than 20
  discarded), within each retained group the observed ML variance of AT
  totals is compared to the local-independence variance
  `sum_i p_i (1 - p_i)` normalized by Stout's standard-error estimate
  `S_k`, giving `T_L = K^{-1/2} sum_k (sigma_k^2 - sigma_U,k^2)/S_k`, the
  AT2 bias correction `T_B`, and `T = (T_L - T_B)/sqrt(2)` with a one-sided
  upper-tail normal p-value. Formulas transcribed from Nandakumar & Stout's
  1992 ERIC technical report ED351383 (published 1993, *Journal of
  Educational Statistics, 18*(1), 41-68), which describes Stout (1987,
  Sec. 4); Kieftenbeld & Nandakumar (2015, PMC5978610) READ for the
  original-vs-bootstrap bias-correction distinction. NOT read: Stout (1987)
  original article, Stout et al. (2001), Froelich & Habing (2008), DIM-Pack
  sources — no ATFIND, no DIMTEST 2 / bootstrap correction, no polytomous
  items, no missing data. Pinned against an independent NumPy oracle
  (500x18 two-dimensional fixture, agreement 1e-12 on `T_L`/`T_B`/`T`;
  p-value at 5e-7 due to the crate's Numerical Recipes `erfc`).

- **Confidence-interval (ACI) classification for CAT**
  (`fast_mlsirm.ci_classify`; in Rust `mlsirm_core::exposure::ci_classify`,
  PyO3 `py_ci_classify`): single-cut binary-response classification by
  interim EAP ability estimate on a fixed 41-point `[-4, 4]` grid with
  standard-normal prior, SE = EAP posterior SD, interval
  `theta_hat +/- z_crit * se` vs `theta_cut` with STRICT first-crossing
  decisions -> `"above"`/`"below"`/`"continue"` with 1-based `n_used`; full
  theta/se/lower/upper traces are returned as offline diagnostics (entries
  past `n_used` are counterfactual replay values). Verified against R catIrt
  `termCI.R`/`eapEst.R`/`catIrt.Rd` at commit
  `c9e979e4812c27d95d367a7f097edfe8e93ac8eb` (READ); Kingsbury & Weiss
  (1983), Thompson (2007), and Eggen & Straetmans (2000) were NOT
  method-section verified and are historical/background context only.
- **Wald SPRT classification for CAT** (`fast_mlsirm.sprt_classify`; in
  `mlsirm_core::exposure`). Single-cut binary-response sequential probability
  ratio test: point hypotheses at `theta_cut -/+ delta`, cumulative binary
  log-likelihood ratio under the D=1 logistic 3PL, and inclusive
  first-crossing decisions against the log Wald boundaries
  `A = ln((1-beta)/alpha)`, `B = ln(beta/(1-alpha))` -> `"above"`/`"below"`/
  `"continue"` with 1-based `n_used`; the full `llr_trace` is returned as an
  offline diagnostic (entries past `n_used` are counterfactual replay
  values). Verified against R catIrt `termSPRT.R`/`logLik.brm.R`/`p.brm.R`
  and Thompson (2007, doi:10.7275/fq3r-zz60); Reckase (1983), Eggen (1999),
  and Wald (1947) are cited as historical origins via Thompson (not directly
  read). Log-likelihood ratios are computed in stable log space (softplus /
  log-sigmoid), so extreme-but-valid parameters that saturate the response
  probability to numerical 0/1 yield finite LLRs instead of errors. Pinned
  17-digit interior-crossing oracle, error-path and 500-rep Monte-Carlo
  structural-invariant tests; 4 executed mutation kills (swapped boundaries,
  dropped guessing floor, collapsed null hypothesis, off-by-one `n_used`).
- **Owen-approximate posterior-predictive EPV item selection**
  (`fast_mlsirm.epv_select`; in `mlsirm_core::exposure`). Deliberately
  reduced scope of van der Linden's (1998, doi:10.1007/BF02294775) minimum
  expected posterior variance (MEPV) criterion: the posterior is Owen's
  normal approximation `N(mu, sig2)`, the predictive probability is
  `p*_i = c_i + (1-c_i) Phi((mu-b_i)/sqrt(1/a_i^2 + sig2))`, and the outcome
  posterior variances come from `owen_update` rather than exact numerical
  posteriors; the unadministered item minimizing
  `EPV_i = p*_i sig2_i^+ + (1-p*_i) sig2_i^-` is selected (lowest-index
  ties). van der Linden (1998) READ as ERIC ED424235 (Research Report
  96-01); the exact-MEPV contract additionally verified against R catR
  `EPV.R` and mirtCAT `selection_criteria.R` (both READ); Owen (1975) NOT
  read (update formulas follow the crate's `owen_update`). Pinned oracles
  and a delegation discriminator (argmin EPV vs. max-info vs. b-matching)
  fixed by the adversarial spec review.
- **Kingsbury-Zara constrained CAT (CCAT) content balancing**
  (`fast_mlsirm.ccat_select`; in `mlsirm_core::exposure`). Single-step
  content-balanced item selection: eligible groups with zero administered
  items have priority, otherwise the eligible group with the maximal
  target-minus-empirical-proportion discrepancy is chosen; within the chosen
  group the unadministered item with maximal logistic 3PL Fisher information
  `a^2 (Q/P) ((P-c)/(1-c))^2` is selected. Ties go to the lowest index
  (documented deterministic deviation from catR's random tie-break).
  Kingsbury & Zara (1989, doi:10.1207/s15324818ame0204_6) itself NOT read
  (paywalled); the rule is implemented as reproduced by the R catR package
  (`nextItem.R` `cbControl` branch; READ), and the information formula was
  verified against catR `Ii.R`/`Pi.R`. Pinned oracles computed in exact
  arithmetic by the adversarial spec review.
- **Owen approximate Bayesian sequential CAT** (`fast_mlsirm.owen_update`,
  `fast_mlsirm.owen_cat`; in `mlsirm_core::exposure`). Closed-form
  normal-approximation posterior moment updates for the 3PNO model
  (`P = c + (1-c)Phi(a(theta-b))`) and a sequential driver with Owen's
  b-matching selection (`argmin |b_i - mu|`, ties to the lowest index) and
  posterior-variance stopping rule (plus a `test_length` cap). Owen (1975)
  itself NOT read (paywalled); formulas implemented as reproduced by
  van der Linden (1998, Research Report 96-01, Appendix A.1-A.6) and
  cross-checked against the R `irt` package `est_ability_owen.cpp`; pinned
  oracles verified against exact-posterior numerical integration (~1e-13)
  by the adversarial spec review.

- **Chang-Ying KL global-information CAT selection**
  (`fast_mlsirm.kl_information`, `fast_mlsirm.kl_select`; in
  `mlsirm_core::exposure`). Kullback-Leibler item index as the UNNORMALIZED
  area of the pointwise Bernoulli divergence (expectation under the
  provisional `theta0`) over `[theta0 - delta, theta0 + delta]` via composite
  Simpson (2048 panels), and next-item selection with the paper's shrinking
  half-width `delta = r / sqrt(n_administered)` (requires `n >= 1`; `r = 3`
  default per Study 1). Administered items keep their computed index; masking
  applies to selection only. Small-delta Fisher limit
  `I(theta0) * delta^3 / 3` anchored by test. Paper READ (Chang & Ying, 1996,
  doi:10.1177/014662169602000303); cross-checked against catR `KL.R`.

- **Raju ICC-area DIF** (`fast_mlsirm.raju_area`; in `mlsirm_core::dif`).
  Parametric signed/unsigned area between two logistic ICCs on a common
  scale, with Raju's delta-method Z tests. Signed area `h = b_F - b_R`
  (positive = harder for focal); unsigned `h = |H|` from Raju's closed form
  via a numerically stable softplus, with a continuous equal-slope fallback
  `|b_F - b_R|`; common-guessing 3PL reports `h`/`se` scaled by `(1 - c)`
  with the Z from unscaled quantities. Primary papers NOT read (Raju 1988,
  Psychometrika 53(4), 495-502, doi:10.1007/BF02294403; Raju 1990, APM
  14(2), 197-207, doi:10.1177/014662169001400208 — both paywalled); formula
  oracle is the difR source (`RajuZ.R`, `difRaju.R`; Magis et al. 2010,
  Behavior Research Methods 42, 847-862, doi:10.3758/BRM.42.3.847 — code
  read in full, package paper not read). Both areas and all four
  delta-method partials re-derived by hand and verified against numeric
  quadrature/finite differences in adversarial spec review; documented difR
  divergences: its gradient is the uniform negation of dH (variance-
  equivalent) and its `exp(Y)==Inf` overflow branch carries the sign
  opposite to the closed-form positive-side limit (unreachable here via
  softplus + `|H|`). Monte-Carlo (500 reps, parametric asymptotic):
  signed test holds nominal level and power; the unsigned Z is measurably
  anti-conservative under an exact equal-slope null (~.14 at nominal .05),
  documented in the API rather than hidden.

- **Velicer minimum average partial (MAP) test** (`fast_mlsirm.velicer_map`,
  `velicer_map_from_data`; in `mlsirm_core::factor`). Component-retention
  test of Velicer (1976, Psychometrika 41(3), 321-327,
  doi:10.1007/BF02293557 — NOT read; formula
  support is the read implementations below): PCA loadings from the
  eigendecomposition of R, partial covariance `C* = R - A_m A_m'` rescaled
  to a partial correlation matrix, and `f2[m]` = mean squared off-diagonal
  partial correlation for `m = 0..max_m` (with the `m = 0` baseline being
  R itself); retained components = the `m` at the minimum. Also computes
  the revised elementwise fourth-power criterion `f4` (Velicer, Eaton, &
  Fava, 2000, in Goffin & Helmes, Problems and Solutions in Human
  Assessment, 41-71 — not read; attributed per O'Connor's code comments).
  Algorithm and retention rule verified against Brian O'Connor's canonical
  MAP programs (map.m and map.sps, oconnor-psych.ok.ubc.ca/nfactors — read
  in full; O'Connor, 2000, Behavior Research Methods, Instruments, &
  Computers 32(3), 396-402, paper itself not read) and psych VSS.R `map()`
  (Revelle, 2025 — read). Documented divergences found by adversarial
  review: `fungible::faMAP` prints a 1-based row position (off by one vs
  O'Connor's count — not reproduced); `EFA.dimensions::MAP` now uses matrix
  powers for the fourth-power criterion, conflicting with O'Connor's
  elementwise form (unresolved from primary literature; we follow
  O'Connor). Rows with singular partial-covariance normalization (e.g.
  identity R for `m >= 1`) are NaN and excluded from the argmin. Rust core
  with PyO3 binding and thin NumPy wrapper; Harman-8 full-vector oracle
  parity (independent NumPy transcription), identity guard, and a
  500-replication Monte-Carlo recovery test (`#[ignore]`).
- **a-stratified multistage CAT item selection** (`fast_mlsirm.a_stratified`;
  in `mlsirm_core::exposure`). Simulation of Chang & Ying's (1999)
  a-stratified design: the pool is split into `n_strata` contiguous strata by
  ascending discrimination `a` (stable sort, near-equal sizes with the first
  `n mod K` strata one item larger — repository choice; catR places the
  remainder last), the test is partitioned into matching stages, and within
  the active stratum the next item is `argmin |b_i - theta_hat|`
  (b-matching, ties to the lowest original index). The b-matching selection
  rule and ascending-a strata are confirmed from Barrada, Mazuela, & Olea
  (2006, Psicothema 18(1), 156-159 — read in full); Chang & Ying (1999,
  Applied Psychological Measurement 23(3), 211-222) is cited as the design's
  origin from its abstract. Interim EAP on a uniform grid and the initial
  `theta_hat = 0` are repository choices (the paper used ML-based interim
  estimation). Returns per-item exposure rates, stratum assignment, stage
  lengths, and theta RMSE/bias; the per-stratum counting identity
  `sum_{i in stratum k} P(A_i) = stage_lengths[k]` holds exactly and is
  regression-tested against returned values. Stratum-level b-blocking
  (Chang, Qian, & Ying, 2001, "a-stratified multistage computerized adaptive
  testing with b blocking", Applied Psychological Measurement 25(4),
  333-341 — not read; excluded per adversarial spec review) is out of
  scope. Rust core with PyO3 binding
  and thin NumPy wrapper; mutation-audited tests plus a 500-replication
  Monte-Carlo comparison (`#[ignore]`) showing lower exposure imbalance
  (summed squared deviation from the uniform rate `L/n`) than
  max-information selection.
- **Sympson-Hetter item-exposure control** (`fast_mlsirm.sympson_hetter`;
  in `mlsirm_core::exposure`). Iterative Monte-Carlo calibration of the
  exposure-control parameters `k_i = P(A_i | S_i)` for dichotomous 3PL
  max-information CAT with interim EAP: per-encounter uniform gate
  (administer iff `u <= k_i`, rejected items blocked for the remainder of
  that simulee's test), update `k_i <- min(1, r_max / P(S_i))` (Barrada,
  Olea, & Ponsoda, 2007, Eq. 1-3; algorithm confirmed from Georgiadou,
  Triantafillou, & Economides, 2007 — both read in full; Sympson & Hetter,
  1985, cited as origin, not read). The stopping rule
  `max P(A) <= r_max + tol` is a practical criterion, not a convergence
  theorem (van der Linden, 2003, abstract); the returned `k` is always the
  vector that produced the reported final-cycle rates. Feasibility bound
  `r_max >= test_length/n_items` (exact counting identity
  `sum_i P(A_i) = test_length`, derived here) is enforced; the bound is
  necessary, not sufficient — a tight `r_max` near the bound may still
  exhaust the pool mid-test and fail with the documented error. `r_max = 1`
  reduces exactly to unconstrained max-info CAT (no exposure RNG
  consumed); an exhausted pool raises an error (repository policy, not a
  classical prescription). Adversarially spec-verified before
  implementation (REDUCED SCOPE: no theta-stratified variants, no
  forced-administration fallback, no "classical iteration count" claim);
  a 500-rep Monte Carlo calibration run (`#[ignore]`); four executed
  mutation kills (gate flip, `P(A)` update denominator, no-blocking —
  killed by divergence/non-termination — and swapped selection/exposure
  bookkeeping).
- **Selection utility analysis** (`fast_mlsirm.selection_utility` /
  `taylor_russell`; in `mlsirm_core::utility`; transcribed from CRAN
  iopsych 0.90.1 `utilityBcg`/`trModel`/`ux`, read in full — Goebl,
  Jones, & Beatty, 2016; the original Taylor & Russell, 1939, Naylor & Shine, 1965,
  and Cronbach & Gleser, 1965 sources were not read and are cited as
  attributed). Formulas under the standard bivariate-normal selection
  model: selection intensity `ux = phi(xc)/sr`, Naylor-Shine selected-group
  criterion mean `pux = rxy*ux`, BCG utility gain
  `n*period*sdy*pux - cost_total` (the iopsych `cost` argument is
  documented here as a TOTAL cost — iopsych labels it per-applicant but
  never multiplies by `n`), and Taylor-Russell success ratio
  `P(Y>yc | X>xc) = Q(xc,yc,rxy)/sr` with the bivariate-normal upper tail
  `Q` evaluated by a conditional-normal Gauss-Legendre integral (~1e-15
  vs scipy's BVN CDF at moderate `|rxy|` during adversarial spec review,
  better than 1e-6 across the whole accepted `|rxy|` range —
  regression-tested; the committed oracle generator is
  `tests/oracles/oracle_utility.py`; the iopsych `qa/(qa+qb)` form was
  proven equal to `Q/sr` algebraically and numerically). Adversarially spec-verified before implementation; five
  scipy-pinned oracle fixtures including negative validity; rho=0
  analytic anchor (success == base rate); strict rxy monotonicity; a
  500-rep x 20,000-person Monte Carlo recovery run (`#[ignore]`; success
  ratio within 4.3e-5, pux within 3.8e-4); four executed mutation kills
  (dropped `rxy` in pux, sign flip in the Q integrand, `1-sr` denominator
  in ux, sr/br role swap). Documented identity limitation: the mutant
  `Q(h,k) -> Q(k,h)` alone is output-identical everywhere by BVN exchange
  symmetry — no test claims to kill it; cutoff-role bugs are anchored by
  the role-swap kill instead. Post-implementation adversarial review
  hardening: sub-ulp ratios (where `1.0 - v` rounds to 1.0) are rejected
  instead of returning NaN/silent zeros; the BVN panel width scales with
  `sqrt(1 - rho^2)` so `|rxy|` near 1 stays accurate (Err beyond
  `sqrt(1-rxy^2) < 1e-4`); `q_joint` is bounded by `min(sr, br)`; all
  three regression-tested against scipy `quad` oracles.
- **Factor-analytic greatest lower bound** (`fast_mlsirm.glb_fa` /
  `glb_fa_from_data`; in `mlsirm_core::factor::glb_fa_corr`; transcribed
  from CRAN psych `glbs.R` `glb.fa`, read in full — Revelle, 2025; NOT the
  algebraic glb of `glb.algebraic`, which requires an SDP solver; Sijtsma,
  2009, not read). Algorithm: 1-factor minres fit, eigenvalues of `R` with
  the diagonal replaced by the model communalities, `nf` = count of
  positive eigenvalues with psych's single df-based decrement, then
  `glb = sum(rr)/sum(R)` with `diag(rr)` from an `nf`-factor refit.
  Verified against a pinned independent scipy oracle on a 9-variable
  2-factor population matrix (glb to 1e-5), a sampled 6-variable matrix
  and a df-adjustment fixture (both df = 0 saturated fits, wider bands
  documented), plus a 500-rep Monte Carlo run (`#[ignore]`; observed mean
  glb 0.863 vs population omega 0.830 — the expected upward bias of glb
  under multi-factor detection is documented, not hidden). Three executed
  mutation kills (skipped diagonal substitution, 1-factor communalities in
  the ratio, dropped df decrement).
- **Person separation reliability** (`fast_mlsirm.separation_reliability`;
  in `mlsirm_core::reliability`; transcribed from CRAN eRm `SepRel.R`, read
  in full — Mair et al., 2025; the statistic is attributed there to Wright
  & Stone, 1999, not read). `R = (SSD - MSE)/SSD` with `SSD = var(measures)`
  (n-1 denominator) and `MSE = mean(se^2)`, unclamped; plus the hand-derived
  separation index `G = sqrt((SSD - MSE)/MSE)` (adjusted true SD over RMSE,
  `G^2 = R/(1-R)`; not in the read source). eRm's extreme-score/NA
  filtering is documented as caller responsibility. Verified against a
  pinned numpy fixture (SSD, MSE, R, G at 12 decimals), a negative-R path,
  and a 500-rep Monte Carlo recovery (`#[ignore]`; population R = 0.8
  recovered to 0.01); three executed mutation kills (swapped numerator,
  population variance, unsquared se).
- **Minres (ULS) exploratory factor analysis and McDonald's omega_total
  (1-factor)** (`fast_mlsirm.minres_fa`, `minres_fa_from_data`,
  `omega_total_1f`, `omega_total_1f_from_data`; in `mlsirm_core::factor`;
  line-by-line transcription of CRAN psych `fa.R`'s minres path — Revelle,
  2025, read; McDonald, 1999, cited-not-read, the omega formula is
  hand-derived from the standardized 1-factor model). Uniquenesses are
  box-constrained to `[0.005, 1]` and optimized by projected
  Barzilai-Borwein descent with an Armijo safeguard and finite-difference
  fallback (psych's `FAgr.minres` direction is not the exact gradient of
  the lower-triangle objective — a verified limitation); convergence is
  certified by a finite-difference box-KKT check whose maximum violation
  is returned (`kkt_violation`). Loadings are unrotated, columns in
  descending-eigenvalue order with column sums >= 0. REDUCED SCOPE: no
  rotation, no Schmid-Leiman / omega_hierarchical, no ML/WLS/GLS, no
  factor scores. Tests pin parity at 5e-5 against an independent scipy
  L-BFGS-B transcription oracle, anchor the absolute objective value of a
  deliberately misfitting 1-factor fit (the only assert that can kill the
  lower-triangle-vs-all-off-diagonal x2 mutation — disclosed), verify
  rank-1 exact recovery and structure invariants, execute four mutation
  kills, and include a 500-rep `#[ignore]` Monte Carlo recovery study.
- **Generalizability theory G/D studies for crossed designs**
  (`fast_mlsirm.gtheory_pi`, `fast_mlsirm.gtheory_pio`; in
  `mlsirm_core::gtheory`; Huebner & Lucht, 2019, read in full — the EMS
  inversions the paper defers to Brennan, 2001, and Shavelson & Webb,
  1991, both cited-not-read, are hand-derived and numerically verified
  against the paper's published Tables 3-6). One-facet `p x i` and
  two-facet `p x i x o` random-effects ANOVA variance components, plus
  D-study relative/absolute error variances, generalizability coefficient
  E-rho^2, and dependability Phi over proposed facet sizes. Negative raw
  components are reported as-is in `var_raw` and clamped to zero in `var`
  for the D study (documented clamped-ANOVA implementation policy);
  coefficients are NaN when their denominator is <= 1e-12. Rust tests
  reproduce the paper's worked examples at full precision, add
  independent RNG-pinned fixtures (including a natural negative-component
  anchor), executed mutation kills (M1/M3/M5/M6), and a 500-rep
  `#[ignore]` Monte-Carlo recovery test.
- **Livingston-Lewis classification accuracy and consistency**
  (`fast_mlsirm.livingston_lewis`; in `mlsirm_core::classification`;
  Livingston & Lewis, 1995, as implemented in Haakstad's CRAN
  `betafunctions` 1.9.0 source `LL.CA` in `R/classification.R`, read line
  by line — the original article was not consulted directly; Hanson, 1991,
  four-parameter beta moment fit, as cited in Haakstad, 2022). From a
  single administration: effective test length
  `((m-min)(max-m) - r s^2)/(s^2 (1-r))`, true-score raw moments via the
  factorial-moment identity on the unrounded-ETL scale, four-parameter
  beta method-of-moments fit with a two-parameter [0, 1] fail-safe, then
  accuracy cells (tp/fp/tf/ff), sensitivity/specificity, consistency
  cells, and Cohen's kappa under a binomial observed-score model with
  `N = round(ETL)`. Integrals use singularity-safe composite
  Gauss-Legendre quadrature (power substitution when a shape parameter is
  below one; endpoint-graded panels otherwise), verified against
  `scipy.integrate.quad` replication literals at 1e-7. Divergences
  (documented in the module): a single round-ties-even threshold
  `k = round(N c)` in both the accuracy and consistency blocks (the oracle
  mixes `round` in accuracy with `floor` in consistency, making its
  consistency cells asymmetric; here `p_ij == p_ji` by construction);
  pass = observed score >= cut is the positive class (the oracle labels
  fail as positive); the fail-safe also engages on numerically invalid
  four-parameter fits (the oracle only checks out-of-bounds support); hard
  errors instead of NA/NaN propagation for invalid inputs, while the
  conditional ratios (sensitivity, specificity, kappa) are an explicit
  `NaN` when their margin or chance denominator vanishes (e.g. a cut
  outside the fitted beta support).
- **Cronbach alpha + Feldt exact-F confidence interval**
  (`fast_mlsirm.cronbach_alpha`, `fast_mlsirm.feldt_alpha_ci`; in
  `mlsirm_core::reliability`; Feldt, 1965, as cited in and implemented by
  Revelle's CRAN `psych` 2.6.5 source `alpha.ci` in `R/alpha.R`, read line
  by line; Cronbach, 1951, covariance form verified against the same
  source). `cronbach_alpha` computes the raw-covariance form
  `p/(p-1) * (1 - tr(C)/sum(C))`; `feldt_alpha_ci` inverts the pivot
  `(1-alpha)/(1-alpha_hat) ~ F(n-1, (n-1)(p-1))` into a two-sided interval
  (`lower = 1-(1-alpha_hat)*qF(1-delta/2)`, upper mirrored) plus the
  implied average inter-item correlation `r_bar`. The F quantile is
  computed in-crate via a Lentz continued-fraction regularized incomplete
  beta and bisection (verified against `scipy.stats.f.ppf` fixture
  literals at 1e-9). Bounds are not clamped; negative alpha is accepted
  into the CI, matching psych. Divergences (documented in the module):
  raw-data input only, zero-variance items rejected, confidence `level`
  argument instead of `p.val`, hard errors instead of NA.
- **ten Berge & Zegers mu reliability series** (`fast_mlsirm.tenberge_mu`;
  in `mlsirm_core::reliability`; ten Berge & Zegers, 1978, as cited in and
  implemented by Revelle's CRAN `psych` 2.6.5 source `tenberge.R`, read
  line by line). On the Pearson correlation matrix with `Vt = sum(R)`,
  off-diagonal power sums `S_k`, and `c = p/(p-1)` on the innermost radical
  only: `mu0 = c*S_1/Vt` (= coefficient alpha = Guttman lambda3),
  `mu1 = (S_1 + sqrt(c*S_2))/Vt` (= Guttman lambda2), `mu2` and `mu3` nest
  one and two further radicals over `S_4` and `S_8`. The series ordering
  `mu0 <= mu1 <= mu2 <= mu3` follows from Cauchy-Schwarz over the `p*(p-1)`
  off-diagonal cells and is asserted on crate outputs. Divergences from
  psych (documented in the module): raw-data input only (no
  correlation-matrix passthrough, no `use = "pairwise"`), hard errors on
  degenerate input, `S_1` summed directly to avoid cancellation. Verified
  against an independent NumPy replication on two fixtures pinned at
  `1e-9`, exact-identity cross-checks against `guttman_lambdas`, and a
  500-replication tau-equivalent Monte Carlo (`#[ignore]`).
- **Guttman lambda reliability coefficients** (`fast_mlsirm.guttman_lambdas`;
  new `mlsirm_core::reliability`; Guttman, 1945, as cited in and implemented
  by Revelle's CRAN `psych` 2.6.5 sources `guttman.R`/`splitHalf.R`/`smc.R`,
  read line by line). On the Pearson correlation matrix: lambda1-lambda3
  (lambda3 = coefficient alpha), lambda5 (best covariance column), lambda6
  (squared multiple correlations via a plain symmetric inverse), plus
  split-half summaries — lambda4 (best split), beta (worst split, floored at
  0), and the mean split over all `C(p, floor(p/2))` subsets when that count
  fits the `n_sample_splits` budget (psych's brute-force cutoff 15000),
  otherwise over LCG-sampled splits. Declared divergences (documented in the
  module): no `check.keys` auto-reversal, absolute split-half correlations
  in both branches (psych's sampled branch is signed), hard error on
  singular correlation matrices instead of psych's pseudoinverse, crate-LCG
  sampling (psych-inspired, not bit-identical to any R run), and duplicate
  sampled subsets allowed. Verified against an independent NumPy replication
  (`np.corrcoef` + `np.linalg.inv` + `itertools.combinations`) on three
  fixtures (even-p exhaustive, odd-p exhaustive, sampled) pinned at `1e-9`,
  plus a 500-replication tau-equivalent Monte Carlo (`#[ignore]`) recovering
  the analytic sum-score reliability within 0.01.
- **Horn's parallel analysis** (`fast_mlsirm.parallel_analysis`; new
  `mlsirm_core::parallel`; Horn, 1965, and Glorfeld, 1995, as cited in and
  implemented by Dinno's CRAN `paran` 1.5.6 sources, read line by line).
  PCA path: eigenvalues of the observed Pearson correlation matrix (cyclic
  Jacobi, eigenvalues only, hard error on non-convergence) are adjusted by
  the sampling bias `random_eigenvalue - 1` estimated from `n_iterations`
  standard-normal data sets of the same shape; components are retained
  while the adjusted eigenvalue stays above 1, scanning left to right and
  stopping at the first failure (later resurgences do not count, matching
  paran's loop-and-break). `centile = 0` uses the per-position mean
  benchmark; `1..=99` uses that upper centile via the R type-7 quantile
  (Glorfeld's conservative variant). Deliberate divergences (documented in
  the module): PCA only (paran's `cfa` generalized-inverse path is out of
  scope), a single deterministic crate-LCG random stream (paran-inspired,
  not bit-identical to any R run), and narrowed guards (`n_persons >= 3`,
  `n_items >= 2`, finite complete data, positive column variance, explicit
  `n_iterations`; the Python wrapper supplies paran's `30 * n_items`
  default). Fixture literals verified against an independent NumPy
  replication that mirrors the LCG stream exactly.

- **IRT classification accuracy and consistency**
  (`fast_mlsirm.rudner_classification`, `fast_mlsirm.lee_classification`; new
  `mlsirm_core::classification`; Rudner, 2001, 2005; Lee, 2010, as cited in
  Lathrop's CRAN `cacIRT` sources). Rudner's normal-approximation method
  treats the observed score at ability theta as N(theta, sem^2) and reports
  per-cut and simultaneous accuracy/consistency, conditional and marginal
  (weights normalized internally; uniform weights reproduce cacIRT's
  person-level `Rud.P`, quadrature weights the distribution-level `Rud.D`).
  Lee's summed-score method replaces the normal approximation with the exact
  Lord-Wingersky (1984) score distribution reused from
  `mlsirm_core::scoring`; raw cuts split scores at `ceil(cut)` and the true
  category is the raw-score interval containing the expected true score.
  Category intervals are left-closed everywhere (cacIRT's `Lee.D` alone is
  right-closed — documented divergence); item probabilities must lie
  strictly inside (0, 1) (rejecting P == 0 is stricter than the oracle,
  which only breaks at P == 1); simultaneous outputs are always populated
  (cacIRT emits them only for two or more cuts). Polytomous Lee, `np.cac`,
  and the MLE/SEM ability helpers are out of scope. Rudner outputs inherit
  the crate `erfc` accuracy (|err| < 1.2e-7); Lee outputs are exact to f64
  rounding. For LLM-as-a-Judge quality management this quantifies how
  reliably a judge's cut score separates pass from fail. Rust-only numerics;
  the Python wrapper validates and marshals. Tests pin marginals and
  conditionals against literals from an independent NumPy transcription
  (exact `math.erf`, own recursion), anchor left-closed cuts with a theta
  exactly on a cut and a dyadic true score exactly on a raw cut, use
  unnormalized weights and a non-integer raw cut as mutation anchors (four
  mutation spot-checks killed), and include a 500-replication ignored Monte
  Carlo ordering long informative tests above short noisy ones.
- **Confirmatory DETECT dimensionality analysis** (`fast_mlsirm.detect_analysis`;
  new `mlsirm_core::detect`; Zhang & Stout, 1999, as cited in Robitzsch, 2024).
  Estimates pairwise conditional covariances of binary items with sum-score
  conditioning — the bias-corrected average of the total-score and pair
  rest-score conditionings, per-group ML covariance aggregated with
  group-frequency weights — and computes the DETECT, ASSI, RATIO, MADCOV100,
  and MCOV100 indices against a known item clustering (labels opaque,
  equality-only). Transcribed line-by-line from the CRAN `sirt` R sources
  (`detect.index.R`, `ccov.np.R`, `ccov_np_compute_ccov_sum_score.R`,
  `conf.detect.R`); matches the explicit `ccov.np(use_sum_score=TRUE,
  scale_score=FALSE)` path — the kernel-smoothed default, missing data
  (sirt pairwise-deletes), sqrt(N)-weighted variants (coincide with
  unweighted under complete data), exploratory cluster search, and polytomous
  DETECT are documented as out of scope. All-zero conditional covariances
  (RATIO `0/0`, NaN in R) are rejected with an error. For LLM-as-a-Judge
  item-quality management this diagnoses whether a rubric partition of judge
  items behaves as distinct dimensions. Rust-only numerics; the Python
  wrapper validates and marshals. Tests pin all five indices and every
  per-pair conditional covariance against literals from an independent NumPy
  transcription (which cannot discriminate the z-standardized default path,
  since unique-value grouping is invariant to monotone transforms — the scope
  statement pins that contract), plus hostile `i64::MIN`/`i64::MAX` labels
  and a 500-replication Monte Carlo (`#[ignore]`) separating 2D simple
  structure from unidimensional data.
- **Haberman subscore added-value analysis** (`fast_mlsirm.subscore_analysis`;
  new `mlsirm_core::subscores`; Haberman, 2008, as cited in Sinharay, 2010).
  For each subscale of a disjoint, exhaustive item partition computes the
  PRMSEs of the three classical-test-theory true-subscore estimators — from
  the observed subscore (`= Cronbach alpha`), from the observed total
  (`rho^2(s_t, x_t) * alpha_x` with the true-score covariance row sum over
  subscore columns only), and from both jointly (Wainer-style augmentation via
  `tau`/`beta`/`gamma`) — plus per-person estimator matrices, the
  `(K+1)^2` score correlation matrix, disattenuated subscore correlations, and
  added-value decisions (Haberman's `PRMSE_s > PRMSE_x`; Sinharay's 2010
  `+ 0.01` margin for augmentation, labeled — CRAN `CTTsub`'s relative rule is
  documented but not implemented). Formulas verified against the Appendix of
  Sinharay (2010, ETS RR-10-16) and the CRAN `subscore` R source read
  line-by-line; degenerate samples (alpha outside `(0, 1]`, zero variance,
  subscore collinear with the total) are rejected instead of propagating NaN.
  For LLM-as-a-Judge item-quality management this decides whether per-domain
  judge subscores add diagnostic value over the overall score. Rust-only
  numerics; the Python wrapper validates and marshals. Tests pin every
  reported statistic against literals from an independent NumPy transcription
  of the R semantics on an asymmetric fixture with mixed added-value
  outcomes, include rejection tests for the structural and degeneracy guards
  (the defensive computed-PRMSE-range guard is not separately exercised), a
  conditional dominance
  sweep on guard-passing random data, and a 500-rep `#[ignore]` Monte Carlo
  MSE comparison; three mutation spot-checks (dropped `m/(m-1)`, rowsum
  including the total column, `tau` numerator sign flip) were run and killed.
- **Kernel-smoothing nonparametric IRT** (`fast_mlsirm.ksirt_analysis`; new
  `mlsirm_core::ksirt`; Ramsay, 1991, as cited in Mazza et al., 2014).
  Estimates option characteristic curves by Nadaraya-Watson kernel regression
  (gaussian/quadratic/uniform kernels) of option indicators on rank-based
  ordinal ability estimates `qnorm(rank/(n+1))`, on an equally spaced
  evaluation grid, with Silverman-rule default bandwidths, plus expected item
  score and expected total score curves. Formulas verified against the
  KernSmoothIRT JSS paper (Mazza et al., 2014, Sections 2-2.3) and the
  KernSmoothIRT R/C++ package source read line-by-line; standard errors and
  cross-validation bandwidth selection are deliberately out of scope (the R
  implementation's SE accumulator is order-dependent and unverifiable from
  read sources). For LLM-as-a-Judge item-quality management this reveals
  non-monotone or poorly discriminating evaluation items without a parametric
  model. Rust-only numerics; the Python wrapper validates and marshals. Tests
  pin a hand-computed 4-person fixture (rank->theta qnorm literals, grid
  endpoints, Silverman constant), enforce structural invariants
  (row-sums-to-one with positive denominators, compact-support zeros,
  zero-denominator fallback), and include a 500-replication Monte Carlo
  recovery study (`#[ignore]`) under normal and skewed ability generation
  using the rank-invariance composition oracle.
- **Mokken scale analysis** (`fast_mlsirm.mokken_analysis`; new
  `mlsirm_core::mokken`; Mokken, 1971, as cited in van der Ark, 2007).
  Computes the Loevinger scalability coefficients `Hij`, `Hi`, `H` and their
  Mokken Z statistics, and partitions items into Mokken scales with the
  automated item selection procedure (AISP, "search normal"), with sample
  statistics and selection mechanics verified line-by-line against the mokken
  R package source (van der Ark, 2007; Straat et al., 2013): `Hij =
  S_ij/Smax_ij` with `Smax` from the comonotone (sorted-column) coupling,
  `Hi`/`H` as ratios of pairwise sums, and per-scale Bonferroni-adjusted Z
  gates. For LLM-as-a-Judge item-quality management this flags evaluation
  items that fail to scale (label 0) and detects multidimensional item pools
  before parametric calibration. Complete integer data required (dichotomous
  or polytomous). Rust-only numerics; the Python wrapper validates and
  marshals. Tests include a brute-force covariance oracle, an exact Guttman
  `H = 1` anchor, a hand-computed Z fixture, a Z-gate anchor whose deletion
  seeds a spurious scale (this test caught a real sign error in the normal
  quantile during development), a hand-constructed Criterion-1 design whose
  negative-`Hij` exclusion is the only active gate (mutation-verified), a
  two-cluster AISP recovery, and an `#[ignore]` 500-replicate Monte Carlo
  (normal + skewed traits).
- **Many-Facet Rasch Model (MFRM) rater-severity calibration** (`fast_mlsirm.fit_facets`;
  new `mlsirm_core::facets`; Linacre, 1989; Eckes, 2015). Fits
  `ln[P(k)/P(k-1)] = theta_p - d_i - c_j - f_k` — the rating scale model
  (Andrich, 1978) with a rater facet — to a `persons x items x raters` array with
  NaN-missing sparse judging plans. For LLM-as-a-Judge calibration this puts each
  judge's severity `c_j` on a common logit scale adjusted for item difficulty and
  respondent ability. Estimation is marginal-ML EM on a Gauss-Hermite grid
  (Bock & Aitkin, 1981), NOT Linacre's JMLE, and the docs say so: estimates match
  the Facets program only up to the JMLE-vs-MMLE difference. Identification:
  `theta ~ N(0,1)`, severities and thresholds centered to sum 0
  (`n_parameters = I + (J-1) + (K-2)`). Reports Linacre's connectedness
  diagnostic via union-find over the person-mediated item∪rater co-observation
  graph; `connected=False` means cross-component severity comparisons rest
  solely on the shared trait prior, not the rating design. Rust-only numerics;
  the Python wrapper validates and marshals. Tests include FD gradient anchors,
  the J=1 RSM-reduction identity, asymmetric-severity recovery, sparse and
  disconnected designs, and an `#[ignore]` 500-replicate Monte Carlo
  (normal + skew-normal traits) bounding severity bias and RMSE; a gradient
  sign-flip mutant was verified to fail 4 tests.
- **Warm's weighted-likelihood ability estimation for POLYTOMOUS items** (`fast_mlsirm.score_wle_poly`;
  new `score_wle_poly` in `mlsirm_core::scoring`; Warm, 1989). The library already had the full
  polytomous model family and polytomous EAP scoring, but its only bias-reduced ML ability estimator was
  dichotomous-only. EAP shrinks toward the population mean, which is exactly what individual score
  reporting must not do, so this closes a real gap rather than adding a third way to do the same thing.
  Solves `dlnL/dtheta + J/(2I) = 0` with `I = sum_k P'_k^2 / P_k` and `J = sum_k P'_k P''_k / P_k`
  accumulated over the person's observed items — the exact generalization of the shipped dichotomous
  `sum_i P' P''/(P Q)`, which is its two-category case. `J` is computed DIRECTLY, never as a derivative
  of `I`. GRM and GPCM; PCM is the GPCM path at `slope = 1`. RSM is deliberately NOT supported and the
  code says why: its fitted `(delta, shared tau)` parameterization is not convertible through any
  exposed API, since `rsm_logprobs` builds the equivalent intercepts internally and does not return
  them.
  **Verification status is stated in the code, because it bounds what may be claimed.** That the
  polytomous Warm correction is `J/(2I)` with `J = sum_k P'P''/P` is confirmed from the `catR` package's
  SOURCE, not from a primary paper — and `catR` keeps its Jeffreys-prior branch as a separate
  expression, so the two estimators are kept distinct here too (Magis & Raîche, 2012). Penfield and
  Bergeron (2005) treat the GPCM but their equations were not obtainable, so nothing here rests on them
  and they are not cited as a source. Separately, and PROVED in-repository rather than taken from a
  source: `J = I'` holds exactly for both shipped families. From `I' = 2J - T` one gets
  `J - I' = -E[l' l'']`, which vanishes because the GPCM's `l''` is category-free and the GRM's sum
  telescopes through `v_0 = v_K = 0`; checked numerically at 80-digit precision against fully numeric
  derivatives (relative `|J - I'| <= 1.1e-30`). The WLE therefore coincides with the Jeffreys modal
  estimate for these two families — but the identity is used ONLY as a test oracle and never as an
  implementation shortcut, because it fails for a graded model with per-boundary slopes and for the 3PL,
  both of which are pinned as negative controls.
  **Numerics.** Per-category quantities are formed division-free from the sigmoids — GPCM
  `P'_k/P_k = a(k - E)`, `P''_k/P_k = (P'_k/P_k)^2 - a^2 Var(k)`; GRM `P'_k/P_k = a(1 - s_k - s_{k+1})`,
  `P''_k/P_k = (P'_k/P_k)^2 - a^2(v_k + v_{k+1})` — so no category probability ever appears in a
  denominator and no probability floor is needed anywhere. The resulting GRM information is algebraically
  identical to the shipped `poly_item_information`, via `v_k - v_{k+1} = P_k(1 - s_k - s_{k+1})`.
  **Both polytomous log-likelihoods are log-concave, yet the weighted objective is genuinely
  multimodal**, because the Warm weight is not: a 3-item GPCM bank in the test suite has stationary
  points at `+0.0988`, `+0.3774` and `+1.3314` while `max lnL'' = -5.6e-5 < 0`, so a solver that brackets
  the first sign change from the left errs by 1.23 logits (2.36 on the GRM fixture). The global grid
  scan of `score_wle` is therefore reused unchanged, including its refusal to return an unresolved mode
  beyond 65,536 intervals; the grid demand additionally scales with `n_cat - 1`, documented as a derived
  worst-case margin for which no wrong-mode counterexample was reproduced.
  **Guards.** Eleven anchors, the important ones mutation-verified with the measured result recorded.
  `J == I'` is pinned with BOTH sides coming from different crate code paths (the accumulator versus a
  central difference of the shipped `poly_item_information`), and the reference magnitudes are pinned to
  1e-5 rather than merely asserted non-zero, so a zeroed or sign-flipped `jterm` fails. `K = 2`
  reproduces the dichotomous `score_wle` for both families — non-discriminating as a design argument,
  but the only anchor that catches a layout transpose, a `cat_params` stride bug or a missing
  chain-rule `a`. The two global-mode fixtures assert the returned value is the dominant mode and not
  the leftmost stationary point; a mutation that stops the scan at the first rise of `Phi` fails eight
  of the ten polytomous tests, and the narrower "take the leftmost stationary point" substitution fails
  the two global-mode fixtures specifically. The estimating equation is re-derived from finite
  differences of the log-probability routines alone, and the all-lowest/all-highest patterns are
  asserted finite alongside a check that the UNWEIGHTED score really does keep a constant sign there,
  so the "the MLE diverges" premise is verified rather than assumed.
  **One coverage limit is stated rather than papered over.** Because `J = I'` is EXACT for both shipped
  families, an implementation that replaced `J` with a numerical derivative of `I` would be
  behaviour-preserving and NO polytomous test can detect it — a mutation confirmed to leave the whole
  polytomous suite green. The discriminating anchors for that substitution live in the dichotomous
  suite, where a lower asymptote breaks the identity. The accompanying test therefore documents that the
  identity is family-specific (exhibiting a per-boundary-slope graded model and a 3PL where it fails,
  with measured relative gaps of 0.92/1.17 and 0.47) and is labelled a lemma about the ORACLE, not a
  test of the code.
  **Also corrects an error in the dichotomous WLE documentation shipped earlier in this release**: it
  claimed `J` coincides with `I'/2` for the 2PL/Rasch. The correct statement is `J = I'` exactly there,
  from `I' = 2J - T` with `T = sum_i P_i'^3 (1 - 2 P_i)/(P_i Q_i)^2`, and `T = J` only when
  `c = 0, d = 1` — which is why the weight is `sqrt(I)`. Fixed in the Rust, PyO3 and Python docstrings;
  the historical entry below is left as written. The identity is now PINNED by
  `wle_information_derivative_identity`, because the first attempt at this correction was itself wrong
  (it dropped the `(1 - 2 P)` factor from `T`, giving a value ~5x off at the 2PL) and nothing caught it.
  A formula asserted in prose and checked by no test is how that happens twice.

- **Uniform SIBTEST, the regression-corrected observed-score DIF procedure** (`fast_mlsirm.sibtest`;
  extends `mlsirm_core::dif`; Shealy & Stout, 1993). The third observed-score DIF procedure in the
  module and the only one that corrects the MATCHING CRITERION itself. Mantel-Haenszel and the logistic
  sweep both match on an observed number-correct score, which is unreliable: under IMPACT — a genuine
  group difference in ability — two examinees from different groups with the same OBSERVED score do not
  have the same expected TRUE score, because each regresses toward their own group's mean. Matching on
  the raw score therefore compares non-equivalent examinees. Item purification, added earlier in this
  release, cannot substitute for this: it changes WHICH items form the criterion, not the regression of
  true score on observed score, so a perfectly purified criterion is still biased. SIBTEST transports
  each group's conditional mean from that group's own Kelley-regressed true score
  `V*_Gk = [Xbar_G + alpha_G (k - Xbar_G)] / n_valid` to the unweighted midpoint of the two, using a
  per-level central difference taken over each group's OWN true-score scale at adjacent OBSERVED level
  positions, and compares the transported means under combined-sample weights renormalized over the
  retained strata. The valid and studied subtests are DISJOINT by construction — the opposite of the
  item-included Mantel-Haenszel default (Donoghue, Holland & Thayer, 1993), and a property of the
  estimator rather than an option. Per-group coefficient alpha is reported on every row because the
  correction divides by it.
  **The headline finding is unflattering and is documented as such.** Measured against
  `mantel_haenszel_dif` on identical simulated data, 500 replications per cell, no DIF planted so every
  rejection is a false positive: at zero impact MH holds .044 and SIBTEST .056; at impact 1.0 MH holds
  .046 while SIBTEST reaches **.086**. SIBTEST over-rejects in both cells and by roughly double under
  impact — the opposite of the ordering its motivation suggests. This is a property of the 1993
  estimator rather than a transcription slip (the closed-form anchors reproduce the TRANSCRIBED FORMULAS
  in exact rational arithmetic; `mirt` itself was never executed, so no cross-implementation agreement
  is claimed), whose standard error treats the ESTIMATED regression correction as fixed and so never
  charges the correction's own noise to the variance. It is precisely what Jiang and Stout's (1998)
  paper — "Improved Type I error control and reduced estimation bias for DIF detection using SIBTEST" —
  was written to fix, and that two-segment estimator is NOT implemented here. The docs accordingly
  recommend Mantel-Haenszel or the logistic sweep for routine screening and position SIBTEST as the way
  to obtain the regression-corrected *estimand*, reading `beta_uni` as an effect size rather than
  trusting `p_value` as a calibrated test. A 500-replication Monte-Carlo test pins that finding so the
  claim cannot rot.
  **Sign.** `beta_uni > 0` means harder for the FOCAL group — the OPPOSITE orientation to `mh_d_dif`
  and `std_p_dif` in the same module, kept rather than harmonised because published `|beta_uni|`
  cut-offs assume it, and asserted in both directions by a cross-module anchor whose product assertion
  would catch a future refactor that "harmonises" both conventions at once.
  **Provenance, stated because it bounds what the code may claim.** Every formula is transcribed from
  the reference implementation (Chalmers, 2012; the `SIBTEST` routine of `mirt`), which attributes them
  to Shealy and Stout (1993); the primary text was not consulted, so no comment cites the 1993 equations
  directly. One deliberate DIVERGENCE from that reference is marked in the code: where a neighbouring
  group-by-level cell is empty it imputes the mean to zero and feeds that fabricated value into the
  central difference, producing a finite but meaningless slope; this implementation drops the level.
  **Scope deliberately reduced after a spec-verification pass.** Crossing-SIBTEST is NOT built:
  Chalmers (2018) shows Li and Stout's (1996) hypothesis test is insufficient, no normal-theory referral
  for a crossing statistic is valid, and `logistic_dif`'s `S x G` interaction already covers crossing
  DIF against a standard 1-df null. No A/B/C letter class, because published cut-offs disagree and none
  was verified against its primary source — the same decision already taken for `delta_r2_uniform`. No
  purified variant, because purification needs a practical-significance predicate that no verified
  cut-off supports, and it would shorten the valid subtest and so lower the very reliability the
  correction divides by.
  **Guards.** Two CLOSED-FORM acceptance anchors, both derived in exact rational arithmetic and
  re-derived independently before the tests were written, assert to 1e-12: a single-stratum fixture
  (`beta = -1/10`, `sigma^2 = 23/1950`, `X^2 = 39/46`) and a five-level fixture pinning the weighting
  (`beta = -16993/88000`). The second is mandatory because the first is structurally blind to every
  weighting question — one retained stratum always carries weight 1 — and because the UNCORRECTED beta
  is `+0.11` under both weighting schemes, so the same assertion on an uncorrected statistic would prove
  nothing. Both were mutation-verified: substituting the observed mean for the true-score subtrahend
  yields `+0.26` (a sign flip), caught by both anchors; focal-group weights yield `-0.039932`, caught
  ONLY by the multi-level anchor. A third anchor pins a NON-CONTIGUOUS level vector to `1/128`, where
  both the hardcoded `2*alpha/n_valid` denominator and arithmetic `k +/- 1` indexing return `1/64` —
  every other fixture has contiguous levels, so without it both mutants survive the whole suite.
  Further anchors pin the strict `j_min` inequality on BOTH sides of its conjunction, all four corners
  of the empty-neighbour guard, the alpha gate on all four degenerate forms, per-group alpha against
  the direct KR-20 definition on a fixture whose groups differ in reliability (a pooled alpha survives
  any fixture where they do not), Benjamini-Hochberg in both directions plus `fdr_q` plumbing, and the
  disjointness of the criterion. That last one asserts the exact invariant rather than a proxy:
  complementing the studied item's column sends every conditional mean to `1 - Ybar` and every slope to
  `-M`, so `beta_uni` is exactly NEGATED and `se_beta` exactly invariant, while at least one other item
  must move — a criterion that ignored the data would be flip-invariant too.

- **Iterative item purification for the observed-score DIF procedures**
  (`fast_mlsirm.mantel_haenszel_dif_purified`, `logistic_dif_purified`; extends `mlsirm_core::dif`;
  Candell & Drasgow, 1988; Clauser et al., 1993; Holland & Thayer, 1988; Lord, 1980). Both DIF
  procedures added earlier in this release match examinees on the observed total score, which is
  itself built from the items under test — so when DIF items push a group's total in a CONSISTENT
  direction, the matching criterion is biased and clean items inherit spurious DIF (a false-positive
  inflation documented at both entry points). Purification breaks that circularity by re-running the
  sweep with the criterion rebuilt from the currently-unflagged ANCHOR set only: round 0 is the
  ordinary all-items sweep, each later round drops the flagged items from the criterion, and the loop
  stops when the flagged set comes back UNCHANGED from one round to the next (`converged = true`) or the
  round cap is hit. That is a stability test, not general cycle detection: a flagged set that oscillates
  between two states runs to the cap and is reported as `converged = false`, which is the honest answer
  rather than a spurious fixed point. The studied item is always added back into its own matching score
  even when it is not in the anchor, so every item is matched on `anchor UNION {studied}` — item-included
  matching is what makes the null-DIF condition hold (Holland & Thayer, 1988; Zwick, 1990), and it also
  makes round 0 identical to the unpurified sweep by construction: round 0 passes no anchor at all and so
  dispatches to the very same code path rather than to an all-true mask that merely evaluates the same.
  `PurifyConfig { max_rounds, min_anchor_items }` bounds the loop and refuses to purify below a usable
  anchor length (default 4), returning the last valid round with `converged = false` rather than a
  criterion built from a handful of items. Nothing is sized from the caller-supplied item count before
  that first sweep, since dimension validation lives inside the sweep and the count is untrusted at the
  FFI boundary.
  **Interpretation limits, documented at the API.** Purification REDUCES contamination, it does not
  remove it: the anchor is itself estimated, so the residual bias depends on how well round 0 separated
  the bank. More importantly the returned p-values carry **no Benjamini-Hochberg or Type-I guarantee** —
  the item set was selected using the same data, so the procedure is a screening device for flagging,
  not a calibrated test; the reported statistics must not be quoted as if they came from a single
  pre-registered sweep. Mantel-Haenszel purification also inherits MH's crossing-DIF blind spot and
  cannot repair it — an item MH never flags stays in the anchor every round and keeps contaminating the
  "purified" criterion. That blind spot is a property of the SIGNED AREA between the two curves over the
  matched ability distribution (Wang & Su, 2004) rather than of non-uniform DIF as such: a crossing at
  the centre of that distribution cancels and is invisible, while the same item with its crossing off
  centre leaves a net difference MH detects, so MH purification is unreliable rather than uniformly blind
  under non-uniform DIF. `logistic_dif_purified`, whose interaction term tests the crossing directly, is
  the variant to use there. **Guards.** A contamination fixture plants unidirectional DIF and asserts, in order, the
  PRECONDITION that the unpurified sweep really does false-flag clean items (so the test cannot pass on
  a fixture with nothing to fix), that purification strictly reduces those false flags, that the true
  positives are retained and are the items that left the anchor, and that the sweep statistics numerically
  changed; a clean bank asserts round 0 reproduces the shipped sweep EXACTLY (`n_anchor == n_items`,
  `rounds == 0`); and the cap and short-anchor exits are each pinned to report `converged = false`
  instead of silently returning a degenerate criterion. An adversarial implementation review then found
  those flag-counting fixtures could not see the arithmetic underneath them, and two structural anchors
  were added, each mutation-verified to fail on the defect it targets. (i) The returned `rows` must equal
  a fresh sweep against the returned `anchor`, swept over round caps, anchor floors and both matching
  conventions (which also covers `exclude_studied_item = true`, previously untested under purification):
  returning an earlier round's rows while reporting the final anchor is the highest-severity failure mode
  of a purification loop and is invisible to a "did it flag the right items" test, because intermediate
  rounds usually flag the same items. (ii) The purified row for an item must equal the ORDINARY sweep run
  on a reduced test consisting of exactly `anchor UNION {studied}` — an independent reference rather than
  the implementation's own arithmetic — checked for both a non-anchor item (the add-back branch) and an
  anchor item, with a deliberately NON-CONTIGUOUS anchor so an index-map error cannot hide behind a
  prefix. The anchor predicate is also pinned directly on all four ETS classes, since no simulated bank
  distinguishes "B or C" from "not A" (a clean 2PL never produces `Undefined`). The same review caught a
  key collision in the Python bindings: `logistic_dif_purified` wrote the loop's scalar convergence flag
  over `logistic_dif`'s PER-ITEM `converged` array, destroying it at the boundary — the loop flag is now
  `purify_converged` on both entry points — and found that both Python docstrings were written as
  `"""..."""` + a module constant, which is a `BinOp` rather than a constant expression, so the compiler
  never filled `__doc__` and the entire "not a calibrated test" caveat was invisible to `help()`.

- **Zumbo logistic-regression DIF, with non-uniform detection** (`fast_mlsirm.logistic_dif`; extends
  `mlsirm_core::dif`; Zumbo, 1999; Swaminathan & Rogers, 1990). Regresses each item response on the
  observed matching score `S`, the group `G`, and their interaction in three NESTED logistic models
  (`M0: b0 + b1 S`; `M1: + b2 G`; `M2: + b3 (S x G)`), fitted by IRLS/Newton. This closes the known blind
  spot of the Mantel-Haenszel procedure added earlier in this release: a stratified odds-ratio test can
  only see a *consistent* group advantage, so **crossing (non-uniform) DIF is invisible to it**, while
  the interaction term detects it directly. The 2-df `chi2_total = 2[ll(M2) - ll(M0)]` is the primary
  omnibus DIF decision (the value Benjamini-Hochberg adjusts); the 1-df components are descriptive
  follow-ups, and the module documents that `chi2_uniform = 2[ll(M1) - ll(M0)]` tests `b2` *assuming*
  `b3 = 0` — it is not the group term of the full model and is uninterpretable when non-uniform DIF is
  present, so the hierarchical entry order `S -> G -> S x G` is load-bearing. The effect size is the
  Nagelkerke (1991) pseudo-`R^2` change `delta_r2 = R2_N(M2) - R2_N(M0)` (with `ll_null` the
  intercept-only fit, and all four models fitted on one identical subsample so the normalizer is
  comparable), classified by Jodoin & Gierl (2001) — A `< 0.035`, B, C `>= 0.070` — and forced to A
  whenever the omnibus test is not BH-significant. The uniform-only `delta_r2_uniform` is reported
  without a letter class because those cut-offs were calibrated on the 2-df quantity; the more
  conservative Zumbo & Thomas (1997) cut-offs are documented as an alternative. **Robustness.** The
  matching score is mean-centered (the chi-squares are invariant, but the raw total leaves the `S x G`
  Gram near-singular); the design is rank-checked; the Newton step uses the *checked* solver and
  step-halving with a coefficient bound, so (quasi-)separation, a rank-deficient design, or
  non-convergence yield `NaN` statistics with `converged = false` and are never BH-flagged (a constant
  item is rejected outright). **Guards.** A SATURATED-DESIGN anchor (two-level score x binary group)
  pins the IRLS, log-likelihood, omnibus chi-square and Nagelkerke effect size against closed-form
  binomial arithmetic, plus the exact decomposition `chi2_uniform + chi2_nonuniform == chi2_total`; and
  the discriminating anchor plants a crossing item whose ICCs intersect at the common group ability
  mean, asserting the logistic test flags the interaction (with the uniform component non-significant)
  on the very item Mantel-Haenszel classifies as negligible, while a plain b-shift item shows the
  reverse pattern; the Jodoin-Gierl classifier is additionally pinned at its boundaries. Spec-verified
  (GO-WITH-MUST-FIXES applied), and an adversarial implementation review then fixed four further root
  causes: a NaN chi-square silently becoming `p = 1.0` in `chi2_sf` (which both misreported "no DIF" and
  let unfittable items dilute the Benjamini-Hochberg denominator), a convergence backstop that certified
  a bound-truncated separated fit as converged, a minimum-sample floor far too weak for the
  four-parameter model, and the unpinned classifier. Convergence uses the standard GLM relative-deviance
  test paired with a coefficient-bound separation check, since near the optimum the attainable score
  floor exceeds any usable absolute gradient tolerance. Same matching-criterion contamination as
  Mantel-Haenszel (see `logistic_dif_purified`), plus the logit-linearity-in-`S` assumption, both documented.

- **Rasch conditional maximum likelihood + Andersen's LR test** (`fast_mlsirm.fit_rasch_cml`,
  `andersen_lr_test`; new `mlsirm_core::rasch_cml`; Andersen, 1970, 1972, 1973). CML estimation of the
  dichotomous Rasch item difficulties: conditioning each response pattern on its raw score (the
  sufficient statistic for ability) ELIMINATES the person parameters, so the difficulties are estimated
  without any assumption on the ability distribution (Rasch's specific objectivity) and consistently at
  fixed test length — unlike the marginal-ML path (which must posit a `theta` distribution) or joint ML
  (inconsistent). The conditional log-likelihood
  `ln L_c = -sum_i s_i beta_i - sum_r n_r ln gamma_r(eps)` uses the elementary symmetric functions
  `gamma_r`; the ESF and its per-item/per-pair derivatives are computed by the numerically stable
  SUMMATION algorithm (a fresh forward pass `gamma_r += eps_j gamma_{r-1}` over the relevant item
  subset), avoiding the cancellation-prone subtractive difference recursion (Verhelst, Glas & van der
  Sluis, 1984). Newton on `beta` with sum-zero identification and a reduced-system solve; standard
  errors from the pseudoinverse of the conditional information; persons scoring `0` or `k` are dropped.
  Andersen's (1973) conditional likelihood-ratio test partitions the persons, fits CML within each group
  and pooled, and refers `2[sum_g llc_g - llc_pooled]` to `chi^2((G-1)(k-1))`. **Guards.** The
  summation-algorithm ESF (and its leave-one-out / leave-two-out passes) match brute-force subset sums;
  a deterministic finite-difference anchor pins the CML gradient AND Hessian (catching the
  `d eps/d beta = -eps` sign); the DEFINING person-distribution-free property is the primary anchor — the
  same `beta_hat` is recovered whether `theta` is `N(0,1)` or strongly right-skewed (a value-recovery
  test alone cannot separate CML from JML); and the Andersen LR does not over-reject Rasch data but
  rejects a planted group-specific difficulty shift, with the `df` and upper tail pinned. Spec-verified
  (GO-WITH-MUST-FIXES applied: the summation ESF over the difference recursion, dropping `r=0/k` persons,
  sum-zero centering, the reduced-Hessian SE, and reuse of `solve_small`/`chi2_sf`). An adversarial
  implementation review (faithfulness clean) then hardened two edge cases: `andersen_lr_test` surfaces a
  `converged` flag so a stalled fit's clamped `lr = 0` is not misread as a clean non-rejection, and the
  Python binding caps `n_groups` at 256 (u8 label range). Complete-data only; polytomous and missing-data
  CML are deferred. Exposed to Python as `fit_rasch_cml` and `andersen_lr_test`.

- **Warm's weighted likelihood estimation of ability** (`fast_mlsirm.score_wle`;
  `mlsirm_core::scoring::score_wle`; Warm, 1989). The bias-reduced maximum-likelihood ability estimator
  for unidimensional dichotomous items (2PL/3PL/4PL): it solves the weighted-likelihood estimating
  equation `dlnL/dtheta + J(theta)/(2 I(theta)) = 0` with the Warm correction
  `J = sum_i P_i' P_i''/(P_i Q_i)` computed DIRECTLY. Crucially `J` is *not* `I'(theta)/2` — the two
  coincide only for the 2PL/Rasch (`c=0, d=1`), where the weight is `sqrt(I)` (the Jeffreys prior); for
  the 3PL/4PL the second derivative carries `1-2s` while the information derivative carries `1-2P`, so a
  `sqrt(I)`-weighted estimator applies the wrong correction. Warm's estimator removes the leading
  `O(1/n)` MLE bias and — unlike the MLE, which is `+/-infinity` for the all-correct / all-incorrect
  pattern — yields a FINITE estimate for every response pattern. The estimate is the GLOBAL maximizer of
  the weighted log-likelihood (whose derivative is the estimating function), located by a grid scan plus
  a local root refinement — robust to the 3PL/4PL case where the weighted likelihood is multimodal
  (Samejima, 1973; Yen, Burket & Sykes, 1991) and a single bracketed root can select the wrong mode;
  it is clamped and flagged when the finite root falls beyond `theta_bound`, and a person with no
  observed items returns `NaN`. It reuses `item_information_4pl` for `I(theta)`; the SE is `1/sqrt(I)`.
  **Guards.** An estimating-equation root anchor is verified by INDEPENDENT finite-difference
  derivatives of `P` (so a `J` sign error in the analytic `P' P''` is not shared) across the 2PL, 3PL,
  and Rasch; a 2PL finiteness anchor confirms the perfect/zero patterns give finite, interior estimates
  with correct > incorrect; a monotonicity anchor confirms the estimate is nondecreasing in the
  number-correct score; a global-mode anchor confirms the multimodal-3PL worst case returns the dominant
  mode (`theta ~ -4.13`, ~10x more probable) rather than a minor root; an all-missing person returns
  `NaN`; and a `#[ignore]` >=500-rep Monte-Carlo confirms Warm's headline result — the WLE aggregate
  `|bias|` (~0.04) is an order of magnitude smaller than the boundary-clamped MLE's (~0.50), the gap
  widening at extreme abilities. Spec-verified (GO-WITH-MUST-FIXES: `J`-not-`I'`, the `I~0` division
  guard, plain natural-scale `a/b/c/d` rather than the log-alpha `ItemBank`); an adversarial
  implementation review then caught and fixed two defects the initial tests missed — the 3PL/4PL
  multimodality (a single bracketed bisection could return a non-dominant root; replaced by the
  global-mode grid search) and an all-missing person silently returning `theta = 0` (now `NaN`).
  Polytomous WLE (Penfield & Bergeron, 2005) is deferred. Exposed to Python as `score_wle` returning
  `theta`/`se`/`boundary`.

- **Mantel-Haenszel differential item functioning** (`fast_mlsirm.mantel_haenszel_dif`; new
  `mlsirm_core::dif`; Holland & Thayer, 1988). The observed-score, calibration-free DIF procedure — the
  complement to the parametric IRT-LR DIF (`dif_polytomous`): no item response model is fitted.
  Examinees are matched on the number-correct total (thin matching, studied item **included** by
  default per Donoghue, Holland & Thayer, 1993; `exclude_studied_item=True` uses the rest score), and
  per item the common odds ratio `alpha_MH = (sum_m A_m D_m / T_m)/(sum_m B_m C_m / T_m)` and the
  continuity-corrected MH chi-square `max(0, |sum A_m - sum E(A_m)| - 0.5)^2 / sum Var(A_m)` (with the
  hypergeometric `Var(A_m) = n_Rm n_Fm m1_m m0_m / (T_m^2(T_m-1))`, referred to `chi^2(1)`) are computed
  over the DIF-informative strata (all four `2 x 2` marginal totals positive). Reported on the **ETS
  delta metric** `MH_D-DIF = -2.35 ln(alpha_MH)` (negative = harder for the focal group) with the
  Robins-Breslow-Greenland (1986) standard error, the **ETS A/B/C** severity classification (Zieky,
  1993; A if not significant at .05 or `|D-DIF| < 1.0`, C if `|D-DIF| >= 1.5` and `|D-DIF| - 1.645 SE >
  1.0`, B otherwise — or `Undefined`/`"U"` when there are no informative strata or a degenerate odds
  ratio, *not* the affirmative "A"), and the **standardized P-DIF** companion (Dorans & Kulick, 1986)
  `sum_m n_Fm (P_Fm - P_Rm) / sum_m n_Fm` (focal minus reference, so its sign agrees with `MH_D-DIF`).
  Benjamini-Hochberg controls the across-item FDR; the p-value reuses `fitstats::chi2_sf`. **Guards.** A
  two-stratum hand-computed anchor pins `alpha_MH`, the continuity-corrected chi-square, `MH_D-DIF`, the
  RBG SE, `STD-P-DIF`, and the C label; a no-DIF symmetry anchor returns `alpha_MH = 1`, zero delta, and
  class A; a degenerate/perfect-separation case returns NaN statistics and `Undefined` (never A); and a
  planted uniform-DIF simulation flags the DIF item (class B/C, correct delta sign) while classifying
  the clean items A and agreeing with the parametric IRT-LR DIF on the flagged item. Because the MH
  chi-square is over-powered at large N and the studied item mildly contaminates the matching total, the
  A/B/C classification (not the raw significance) is the practical-significance guard — documented, with
  item purification (since shipped as `mantel_haenszel_dif_purified`) and SIBTEST (Shealy & Stout, 1993)
  noted as future work. Spec-verified
  (GO-WITH-MUST-FIXES: STD-P-DIF sign, `Var_m > 0` stratum gate, degenerate-odds guards, zero-clamped
  continuity numerator).

- **Dimension-agnostic IRT model API.** Item families are named by their
  response function rather than by UIRT/MIRT dimensionality:
  `fit_2pl`/`TwoPlFit`, `fit_grm`/`GrmFit`, and
  `fit_nominal`/`NominalResponseFit`. A single `model=` argument follows the
  R `mirt` convention (Chalmers, 2012): `model=1` denotes the unrestricted
  one-factor model, while `model=models.confirmatory(loading_pattern)` carries
  a confirmatory loading structure and derives its dimension count. The fitted
  result retains `n_dims` only as a derived read-only property of its model
  specification. Numeric exploratory requests above one factor fail explicitly;
  the Rust estimators do not yet implement unrestricted multidimensional loading
  rotation/identification, so a confirmatory anchor pattern is never relabeled
  as exploratory. The previous brand-new `*_mirt` entry points and module names
  were removed rather than retained as misleading aliases. See
  `python/fast_mlsirm/models.py` for the verified Chalmers (2012) APA reference
  and DOI.

- **Correlated latent factors for MH-RM** (`fit_mhrm(..., estimate_corr=True)`; Cai, 2010b confirmatory
  item factor analysis). Completes the MH-RM to a free latent CORRELATION matrix `Phi` (unit diagonal,
  `theta ~ MVN(0, Phi)`) rather than orthogonal factors. The Metropolis acceptance prior becomes
  `-0.5 (theta*^T Phi^{-1} theta* - theta^T Phi^{-1} theta)` (the symmetric proposal cancels; `Phi^{-1}`
  is recomputed by Cholesky each cycle), and the `D(D-1)/2` free off-diagonal correlations ascend the
  Gaussian-prior objective `Q(Phi) = -0.5[log|Phi| + tr(Phi^{-1} C)]` (`C` the imputed second moment,
  RAW/uncentered — `E[theta]=0` is fixed by identification) by a per-cycle Robbins-Monro GRADIENT step
  `offdiag += gain_k * sigma_grad(Phi, C)`, kept positive-definite by BACKTRACKING (halve the step
  until the rebuilt `Phi` is PD). This REUSES the `twopl.rs` correlation machinery verbatim
  (`build_corr`, `sigma_grad`, `chol_lower`, `sym_inv_logdet`, `flip_corr_dim`, now `pub(crate)`), the
  same helpers `fit_2pl`'s deterministic ECM correlation step uses — so the `Phi` estimation is shared,
  not duplicated. The per-dimension reflection flips the correlation off-diagonals for the flipped
  dimension (`corr(theta_d, theta_k) -> -corr`) together with the loading column and trait chain, so
  the reported `Phi` is consistent with the canonicalized signs. `estimate_corr=False` (default) keeps
  `Phi = I` and is BIT-IDENTICAL to the previous orthogonal fit (the acceptance prior branches to the
  original per-dimension `||theta*||^2 - ||theta||^2` on the same RNG stream). It is a gradient-RM (not
  Cai's Newton-preconditioned) covariance update — documented as such; it still converges almost surely
  to the same `Phi` root, only the (un-curvature-adapted) rate differs. **Guards.** A recovery test
  recovers an exchangeable `Phi` off-diagonal at a POSITIVE (`rho=0.4`), a near-PD-boundary
  (`D=3, rho=0.5`), and a NEGATIVE (`rho=-0.5`) correlation within Monte-Carlo tolerance, confirming the
  recovered matrix stays a valid PD correlation matrix; `estimate_corr=False` yields exactly the
  identity; and a `#[ignore]` 500-rep Monte-Carlo at the near-boundary `D=3, rho=0.5` reports the
  correlation RMSE/bias and would surface a persistent PD-backtracking stall. Exposed to Python as the
  `estimate_corr` argument and the `corr` field of `MhrmFit`.

- **Polytomous (GPCM) response family for MH-RM** (`fit_mhrm(..., family="gpcm", n_cat=K)`; Muraki,
  1992, generalized partial credit model estimated by the Cai, 2010 MH-RM). Extends the
  stochastic-approximation confirmatory item factor analysis from binary items to ordered polytomous
  items, scaling high-dimensional POLYTOMOUS IFA to a latent dimensionality where the deterministic
  `fit_gpcm` (Gauss-Hermite / QMC EM) is infeasible. Each item keeps a SINGLE multidimensional
  discrimination `a_i` (free on the confirmatory loading pattern) and gains `K-1` free UNORDERED step
  intercepts: `base_i = sum_{d in S_i} a_id theta_d` (NO intercept), `P(Y=k) = softmax_k(k*base_i +
  step_ik)` (`step_i0 = 0` pinned). The MH imputation likelihood is the inline log-softmax of the
  observed category (no per-node allocation), and the per-item RM step uses the **closed-form
  multinomial complete-data Hessian** `H = sum_p J_p^T (diag(P) - P P^T) J_p` (data-independent given
  `theta`, where the design row `J_p[k]` is `d psi_k / d param`: `k*theta_pd` for slope `a_id`, `[k==j]`
  for `step_j`) as BOTH the Robbins-Monro preconditioner AND the Louis positive term — NOT the BHHH
  score cross-product (which is the term Louis subtracts, so `H_BHHH - sum s s^T = 0` would give a
  degenerate SE). The complete-data score `sum_p J_p^T ([k==y_p] - P)` equals the deterministic
  `gpcm.rs`'s `[g_base*theta_d; g_intercepts]` with the integer scores fixed (`g_scores` dropped — what
  makes it GPCM, not nominal). The per-dimension reflection flips only the slope column and the trait
  chain — the UNORDERED steps are left INVARIANT (`base = k*sum a_d theta_d` is invariant under the
  joint `(a, theta)` sign flip), exactly as the deterministic `gpcm.rs`. `family="2pl"` (default) keeps
  the binary path **BIT-IDENTICAL** (the closed-form `log_sigmoid` score and `sum w X X^T` information
  are unchanged on the same RNG stream). GRM (Samejima cumulative-logit, ordered thresholds) is
  DEFERRED: its thresholds must stay strictly decreasing, which the deterministic `grm.rs` maintains by
  a backtracking line search a single stochastic RM Newton step cannot replicate (the standard path is a
  softplus threshold-gap reparametrization — future work). An adversarial implementation review found
  and fixed two defects the initial tests missed: the output/SE routing keyed on `n_free_cat == 1` as a
  "is 2PL" proxy, which mis-collapsed a legal `Gpcm { n_cat = 2 }` fit's single step into the 2PL
  `intercept`/`se_intercept` fields (now keyed on the model family); and the declared `MHRM_MAX_CAT`
  category cap was never enforced (an unbounded `n_cat` allocation vector — now validated). **Guards.** A
  deterministic finite-difference
  anchor pins the GPCM score AND the exact-multinomial information against the complete-data GPCM
  log-likelihood on an asymmetric cross-loader with a NEGATIVE loading and NON-MONOTONE steps (a sign
  flip, a transposed/dropped design slot, an over-collapsed step block, or BHHH-as-information all fail
  it), with an independent per-person score outer-product re-sum pinning the sign of the Louis
  missing-information subtraction; a `D=1` reduction test agrees with `poly::fit_poly_unidim(Gpcm)`
  (Bock-Aitkin quadrature) within Monte-Carlo tolerance; a `D=5` recovery (GH/QMC infeasible) recovers
  loadings, steps, and the negative cross-loader with correct sign; a reflection-FIRES test witnesses
  the canonicalization flipping a negative anchor while leaving the steps un-swept; the validation
  rejects out-of-range responses and any never-observed category (an unidentified step); and a
  `#[ignore]` 500-rep Monte-Carlo (normal + right-skew traits, `D=2` and `D=5`, `K=3`) reports the
  loading/step RMSE and bias. Exposed to Python as the `family`/`n_cat` arguments and the
  `step`/`se_step`/`n_cat` fields of `MhrmFit`.

- **High-dimensional confirmatory 2PL by Metropolis-Hastings Robbins-Monro** (Cai, 2010).
  `fit_mhrm(responses, model=...)` fits the general compensatory multidimensional 2PL
  (`P(X_ij = 1 | theta_j) = sigmoid(sum_{d in S_i} a_id theta_jd + b_i)`, `theta ~ MVN(0, I_D)`) — the
  same model as `fit_2pl` — by a STOCHASTIC-approximation EM that scales to a latent dimensionality
  where the deterministic `q^D` Gauss-Hermite grid and the QMC E-step of `fit_2pl` are infeasible
  (`n_dims` up to 64). Each cycle (1) IMPUTES each person's `theta` by a short PERSISTENT
  (warm-started) symmetric random-walk Metropolis chain from its current posterior
  `pi_j(theta) prop phi(theta; 0, I) prod_i P_i(y_ij | theta)` — the acceptance ratio is the pure
  Metropolis posterior ratio (the symmetric proposal cancels), the proposal SD is auto-tuned toward a
  target acceptance during burn-in, and the chain carries across cycles so no per-cycle burn-in is
  needed; and (2) takes one Robbins-Monro stochastic-Newton step
  `xi <- xi + gain_k Gamma_k^{-1} s_k` on the complete-data score `s_k` (Fisher's identity gives an
  unbiased-in-the-limit Monte-Carlo estimate of the marginal score) and the RM-smoothed information
  `Gamma_k = Gamma_{k-1} + gain_k (H_k - Gamma_{k-1})`. Because the item blocks are conditionally
  independent given `theta`, the score, information, and RM step are BLOCK-DIAGONAL by item, and the
  per-item work is the CLOSED-FORM logistic gradient `X'(y - P)` and information `X'WX` — no
  quadrature, `D`-independent per-node cost (reusing `mmle::{log_sigmoid, sigmoid_stable}` and
  `poly::solve_small`). The gain follows a constant-gain burn-in (a Metropolis-Hastings stochastic EM
  that random-walks into the MLE neighbourhood) then a decreasing `gain_k = 1/(k - k0)^alpha`
  (`sum gain = inf`, `sum gain^2 < inf`, Robbins & Monro 1951) that converges almost surely to a
  marginal-score root. Convergence is WINDOWED (the running mean of `||xi^(k) - xi^(k-1)||` over the
  last `w` cycles falls below `tol`) — MH-RM iterates are non-monotone by design, so no
  likelihood-decrease guard is used. **Identification.** Unit trait variances fix the loading scale,
  `E[theta] = 0` the intercepts, and a PURE single-dimension anchor item per dimension pins the
  rotation; the per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` is likelihood-
  invariant, and because the stochastic iterates could otherwise drift between the two mirror modes
  and corrupt the RM RUNNING AVERAGE of the loadings, the canonical sign (largest pure anchor
  positive) is enforced IN-LOOP every cycle — flipping the loading column, the persistent `theta`
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
