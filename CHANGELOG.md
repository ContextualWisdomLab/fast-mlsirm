# Changelog

## Unreleased

### Security

- **Input-validation hardening at the untrusted boundaries** (Strix scan
  findings on PR #160). All are denial-of-service / data-poisoning guards for
  a library that may be exposed as a scoring/fitting service:
  - Population labels (`group_id`/`cluster_id`) are now validated and
    **compacted to contiguous ids** in `fit.py` and `inference.py`, so the
    group/cluster count is the number of *distinct* labels (≤ `n_persons`)
    rather than `max(label)+1` — sparse ids like `[0, 1e9]` no longer force
    billion-row population allocations. Negative, non-integer, non-finite, and
    wrong-length labels are rejected.
  - `FitConfig.validate()` bounds `latent_dim` (≤ `MAX_LATENT_DIM = 8`),
    `xi_points` (≤ `1_000_000`), `max_iter` (≤ `100_000`), `n_restarts`
    (≤ `1_000`), and `m_steps` (≤ `1_000`), and rejects **non-finite**
    `learning_rate`/`init_gamma`/`eps_distance`/`tolerance`/`gradient_clip`
    (a bare `x <= 0` comparison lets `NaN`/`Inf` through) — blocking both
    memory/CPU exhaustion from extreme sizes and NaN-poisoned fits.
  - `plausible_values` bounds `n_draws` (1..`MAX_DRAWS = 100_000`), and
    `serving_prior` bounds `n_dims` (1..64) for direct callers.
  - `load_serving_bundle` parses JSON in **strict mode** (rejects `NaN`/
    `Infinity` literals) and runs a full `_validate_bundle` structural +
    finiteness check (consistent `n_items`/`n_dims`/`latent_dim`, bounded
    sizes, in-range `factor_id`, finite `alpha`/`b`/`zeta`/`tau`/`eps_distance`,
    supported quadrature); `score_respondents` and `plausible_values` validate
    the bundle at entry, so oversized dimensions (e.g. `n_items = 1e12`) and
    non-finite parameters can no longer trigger multi-terabyte allocations or
    NaN scores.
  - `plausible_values` now enforces the binary response domain (0/1, finite)
    that `score_respondents` already required.
  - `validate_judge` validates judge/human/baseline/subgroup labels (1-D,
    equal length, finite, integer, `0 ≤ label < k`) **before** the `uint32`
    conversion, instead of silently truncating floats or wrapping negatives.
  - Regression tests in `tests/test_security_hardening.py` cover each finding.
- **Second-pass hardening** (Strix re-scan of PR #160, 11 findings) extends the
  same DoS/data-poisoning guards to the paper-feature surface added in this PR:
  - `preprocessing.irtree_expand` bounds the dense expansion
    (`persons * items * nodes ≤ 50_000_000`) before allocating, and validates
    `node_dims` (finite, non-negative, integer-valued) before the `int64` cast.
  - `validation._validate_labels` rejects labels above `uint32` max before the
    narrowing cast, and `validate_judge` requires the `human_human` baseline to
    match the paired sample size.
  - `inference.observed_information` caps the finite-difference Hessian at
    `5_000` parameters (it is `O(n²)` memory **and** `O(n²)` objective calls),
    and `oakes_standard_errors` validates `factor_id` (1-D, one-per-item,
    finite, non-negative, integer) before deriving `n_dims`.
  - `serving._validate_bundle` rejects tensor Gauss-Hermite grids that would
    allocate `q_xi ** latent_dim > 1_000_000` points; `estimators.marginal`'s
    `_xi_grid` carries the same bound for direct callers.
  - `linking.link_fixed_item_parameters` rejects duplicate/fractional/negative/
    non-finite anchor indices, non-2-D `theta`, non-finite item parameters, and
    non-finite computed linking coefficients.
- **Third-pass hardening** (Strix re-scan of `b5d9d90`, 11 real findings; the
  12th — "incomplete package release" — was a scanner artifact of its
  PR-scope-only checkout, verified: every named module exists and
  `import fast_mlsirm` succeeds) **plus a proactive boundary audit** that found
  6 more Python issues Strix had not surfaced:
  - `serving.score_respondents`/`plausible_values` bound the dense respondent
    matrix (`rows x n_items`); `serving._validate_bundle` now bounds the
    scoring-table product (`max(n_items, n_dims) x q_theta x q_xi**latent_dim` —
    a 55+ GB allocation otherwise) and validates the bundle `population` block
    (`serving_prior` computed `sqrt(1 + sigma_u**2)` on an unvalidated, fully
    attacker-controlled `sigma_u` → `TypeError`/`OverflowError` crash or silent
    `Inf`/`NaN` score poisoning).
  - `linking.link_fixed_item_parameters` range-checks anchor indices on the
    float **before** the `int64` cast (`uint64` max silently wrapped to `-1`,
    selecting the last item) and requires a positive linking scale;
    `linking.irt_link` validates slope/intercept finiteness and slope
    positivity before the Nelder-Mead core (a `NaN` would panic it).
  - `validation.validate_judge` bounds the category count `k` (drives a dense
    `k x k` confusion matrix) and **compacts** sparse `subgroup` labels (the
    core loops `0..max(label)+1`, an O(4e9) CPU-DoS from one sparse id).
  - `preprocessing.irtree_expand` switched from a 50M-element ceiling (400 MB,
    boundary-inclusive) to a 64 MiB byte budget; `config.MLS2PLMConfig.validate`
    bounds simulation dimensions and the `n_persons x n_items` cell product;
    `config.FitConfig.validate` bounds aggregate optimizer work
    (`max_iter x n_restarts`); `estimators.marginal.fit_marginal_numpy` bounds
    declared population counts (`n_groups`/`n_clusters <= n_persons`) and the
    EM working set; `inference.observed_information` rejects non-finite `step`;
    `inference.oakes_standard_errors` and every `fitstats` public entry bound
    `n_dims` derived from an untrusted `factor_id` (a shared `_validate_factor_id`
    guard); `fit.py` validates anchor/covariate array shapes and finiteness
    before the Rust marginal core.
  - Rust-core backstops for the same audit (defense in depth, active once the
    extension is rebuilt): `fitstats::s_x2` rejects non-dichotomous observed
    responses (a non-0/1 value indexed the summed-score table out of bounds →
    panic); `fitstats::infit_outfit` validates `theta`/`xi` lengths before
    indexing; `scoring::validate_prior` rejects non-finite prior `mean`/`sd`
    (a `NaN` `sd` passed the bare `sd <= 0` check).

### Added

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
  `0 <= p_il <= 1` holds for free (`0 <= R_il <= I_il`), and the all-mastered class
  having the highest success probability is asserted as an invariant rather than
  projected (matching de la Torre's unconstrained-in-`[0,1]` saturated MLE).
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
