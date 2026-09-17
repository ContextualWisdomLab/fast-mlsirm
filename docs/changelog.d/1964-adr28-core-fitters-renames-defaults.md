# ADR-0028 naming/defaults applied to core IRT fitters (#1964)

## Deprecated

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

## Changed

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
