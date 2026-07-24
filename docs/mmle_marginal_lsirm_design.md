# Model-design PR: Marginal (MMLE) estimation for latent-space models, with multigroup and multilevel extensions

Status: implemented by this PR. Paper basis is compiled in
[`docs/papers/mmle-lsirm-formula-compilation.md`](papers/mmle-lsirm-formula-compilation.md)
(per-equation verification legend inside); primary sources: Jeon, Jin, Schweinberger & Baugh (2021,
Psychometrika, doi:10.1007/s11336-021-09762-5), Bock & Aitkin (1981, doi:10.1007/BF02293801),
Bock & Zimowski (1997, doi:10.1007/978-1-4757-2691-6_25), Fox & Glas (2001, doi:10.1007/BF02294839),
Orlando & Thissen (2000, doi:10.1177/01466216000241003), Snijders (2001, doi:10.1007/BF02294437),
Cai (2010, doi:10.1007/s11336-009-9136-x).

## 1. Scope

This PR generalizes the existing `estimator="mmle"` path (previously unidimensional 2PL only,
`crates/mlsirm-core/src/mmle.rs`) to a **marginal EM estimator for all five model variants**
(`MIRT`, `MLS2PLM`, `MLSRM`, `ULS2PLM`, `ULSRM`) under the repo's simple-structure contract

```text
eta_pi = a_i * theta_p,d(i) + b_i - gamma * ||xi_p - zeta_i||,   a_i = exp(alpha_i), gamma = exp(tau)
```

and adds two estimation-level population structures (previously only post-hoc diagnostic strata):

- **Multigroup** (Bock–Zimowski): group-specific trait distributions
  `theta_pd ~ N(mu_gd, sigma_gd^2)` with common (anchored) item parameters and a fixed reference
  group `mu_1d = 0, sigma_1d = 1`.
- **Multilevel** (Fox–Glas random intercept): `theta_pd = sigma_u * u_c + e_pd`,
  `u_c ~ N(0,1)` shared across the trait dimensions of persons in cluster `c`,
  `e_pd ~ N(0,1)`; `sigma_u` estimated (ICC = sigma_u^2 / (1 + sigma_u^2)).

It also adds likelihood-based fit statistics (S-X², l_z, l_z*) and an item-screening pipeline
(`docs/papers/mmle-lsirm-formula-compilation.md` §7–§9), and a serving-bundle export for scoring
new respondents with frozen item parameters.

## 2. Marginal likelihood

Person latents are random effects; item quantities are structural parameters
(formula compilation §3.A):

- `theta_p ∈ R^D`, independent `N(mu_gd, sigma_gd^2)` per dimension (defaults `N(0,1)`),
- `xi_p ∈ R^K ~ MVN_K(0, I)`,
- structural: `alpha, b, zeta, tau` (+ `mu_g, sigma_g` per non-reference group, or `sigma_u`).

```text
L = prod_c  ∫ phi(u) prod_{p in c} L_p(u) du            (multilevel; u vanishes when sigma_u = 0)
L_p(u) = ∫_{R^K} phi_K(x) prod_d [ ∫ phi(t) prod_{i in d, obs} P_pi^y (1-P_pi)^{1-y} dt ] dx
```

with `theta_pd = mu_gd + sigma_gd * t + sigma_u * u` inside `eta`. The **key tractability point**:
under simple structure the trait dimensions are conditionally independent given `xi_p`, so the
integral costs `Q_xi^K * (Q_u) * sum_d Q_theta` — NOT `Q^{1+D+K}`. For the supported `K ≤ 2` this
makes deterministic Gauss–Hermite quadrature feasible; the curse-of-dimensionality warning in the
formula compilation (§3.A.2) applies to the unrestricted model, and MH-RM (Cai 2010) remains the
documented alternative if `K ≥ 3` support is ever needed. A deterministic E-step is also what makes
the Rust↔NumPy 1e-6 parity contract testable, which a stochastic MH-RM path would break.

## 3. EM algorithm

Quadrature: probabilists' Gauss–Hermite, weights normalized to sum 1 (same convention as the
existing `mmle.rs` table). Defaults: `Q_theta = 21`, `Q_xi = 11` per latent axis (tensor grid,
`11^K` nodes), `Q_u = 15`.

**E-step.** Item-response tables are person-independent:
`logP1_i(t, x, s) = log sigmoid(eta_i(t, x, s))`, where `s` indexes the population context
(group `g`, or u-node `v`; absent in the single-population case). Per person:

```text
l_pd(t | x, s)   = sum_{i in d, obs} [ y * logP1 + (1-y) * logP0 ]
logL_pd(x, s)    = logsumexp_t [ log w_t + l_pd(t|x,s) ]
logL_p(s)        = logsumexp_x [ log w_x + sum_d logL_pd(x, s) ]
```

Binary sparsity trick: `l_pd(t|x,s) = C_d(t,x,s) + sum_{i in d: y=1} delta_i(t,x,s)` where
`C_d = sum_{i in d} logP0` is shared and `delta_i = logP1 - logP0`, so the person pass scales with
the count of positive responses, with per-cell corrections for missing entries.

Posteriors (given cluster coupling): `post_c(v) ∝ w_v prod_{p in c} L_p(v)` at cluster level, then
`post_p(t, x | v)` person-level. Expected counts (Bock–Aitkin "artificial data"):

```text
nbar_d(t,x,s) = sum_p post_p(t,x,s)          r_i(t,x,s) = sum_{p: y_pi=1} post_p(t,x,s)
nbar_i = nbar_{d(i)} - (corrections for cells missing on item i)
```

**M-step.**
- Per item `i`: Newton/gradient ascent on the expected binomial log-likelihood
  `sum_{t,x,s} [ r_i log P_i + (nbar_i - r_i) log(1-P_i) ]` over `(alpha_i, b_i, zeta_i)`
  (only `b_i` for `MLSRM/ULSRM`; no `zeta_i` for `MIRT`), with the L2 penalties of
  `PenaltyConfig` (MAP-flavored MMLE; keeps sparse items finite — cf. lognormal slope priors in
  the 2PL LSIRM package, compilation §2.2).
- Global `tau`: 1-D Newton on the same expected log-likelihood.
- Multigroup: `mu_gd = E_g[theta_pd]`, `sigma_gd^2 = E_g[(theta_pd - mu_gd)^2]` from posterior
  moments; reference group pinned.
- Multilevel: `sigma_u^2 <- (1/C) sum_c E[u_c^2 | Y]`.

Convergence: absolute change of marginal log-likelihood `< tol` (same contract as `mmle.rs`).

## 4. Identifiability

- Translation/scale of `theta`, `xi`: fixed by the `N(0,1)` / `MVN(0,I)` population distributions
  (compilation §1.4).
- Rotation/reflection of the latent space: `zeta` is identified up to orthogonal maps. For
  deterministic, run-to-run comparable output the fitted `zeta` (and EAP `xi`) are PCA-aligned:
  rotate so the principal axes of `zeta` coincide with the coordinate axes, sign-fixed so each
  axis's largest-|coordinate| item is positive. Procrustes to an external reference remains the
  documented option for cross-fit comparisons.
- Multigroup: common item parameters are the anchor; reference group `N(0,1)` pins the scale
  (Bock–Zimowski). Multilevel: `E[u]=0`, `Var(e)=1` pin the intercept scale.

## 5. Fit statistics (new `diagnostics` additions)

- **S-X² (Orlando & Thissen 2000)** per trait dimension: summed score over the items of dim `d`;
  `E_is` via the Lord–Wingersky recursion evaluated on the joint `(t, x)` grid with prior weights
  (compilation §7.1); score groups collapsed to expected count ≥ 1; p-values from
  `chi2(df = #collapsed groups - m_i)`, `m_i = 2 + K` for `MLS2PLM` (per-model from exec flags).
- **l_z and l_z\* (Drasgow et al. 1985; Snijders 2001)** per person and trait dimension, evaluated
  at EAP `theta` with `xi` fixed at its EAP (documented approximation; MAP-case correction
  `r_0(theta) = -theta` for the `N(0,1)` prior), formulas exactly as compilation §8.
- Existing infit/outfit MNSQ statistics are reused unchanged.

## 6. Item screening pipeline (compilation §9)

`select_items()` iterates: fit → flag → drop → refit, with an audit trail. Flags per round:

1. pre-screen: fewer than `min_positive` (default 20) positive (or negative) responses;
2. S-X² significant after Benjamini–Hochberg FDR (q = .05);
3. infit/outfit outside `[0.7, 1.3]` (Wright & Linacre 1994 working band);
4. low discrimination `a_i < 0.35` (2PL variants);
5. map isolation: `gamma * mean_p ||xi_p - zeta_i||` a robust-z outlier (> 3) among items.

An item is removed when it fails ≥ 2 of flags 2–5 (or flag 1 alone); persons with
`l_z* < -1.645` are excluded from the *flagging statistics* (not from the final fit). The loop
stops when nothing is removed or a floor (default 4 items per dimension) is reached.

## 7. Serving bundle

`export_serving_bundle(...)` writes a single JSON (plus optional `.npz` mirror) containing:
schema version, model/config, item codes, `alpha/a, b, zeta, tau/gamma`, population parameters
(`mu_g, sigma_g` per group, `sigma_u`), quadrature spec, and the item screening audit. A
`score_respondents(bundle, responses)` function (and `fast-mlsirm score` CLI) computes EAP
`theta`, `xi`, standard errors, and `l_z*` for new response vectors with item parameters frozen —
the same fixed-parameter scoring pattern as the downstream importance-assessment API (mirt
`mod2values`-style freeze, Chalmers 2012).

## 8. Implementation layout & parity

- `crates/mlsirm-core/src/marginal.rs` — f64 CPU reference: quadrature tables, E-step, M-step,
  multigroup/multilevel contexts (single-threaded, deterministic).
- `crates/mlsirm-core/src/gpu.rs` — new f32 WGSL entry points for the person pass and
  expected-count reduction (race-free slot-ownership pattern, same as the JML kernels); `~1e-4`
  agreement, CPU fallback when no adapter.
- `crates/fast-mlsirm-py/src/lib.rs` — `fit_mmle_marginal(...)` PyO3 wrapper.
- `python/fast_mlsirm/estimators/marginal.py` — NumPy mirror (the parity reference), same
  quadrature tables bit-for-bit.
- `python/fast_mlsirm/fit.py` — `estimator="mmle"` routes latent-space/multidim models (and
  `group_id=`/`cluster_id=`) to the new path; the legacy `fit_mmle_2pl` fast path is kept for
  plain `ULS2PLM/ULSRM` without grouping.
- Tests: `tests/test_marginal_parity.py` (Rust↔NumPy 1e-6 on marginal loglik + E-step moments +
  fitted params on small fixtures), `tests/test_estimator_marginal.py` (simulate→fit recovery for
  each variant and each population structure), `tests/test_fitstats.py` (S-X² against hand-computed
  small cases; l_z* properties), Rust unit tests in `marginal.rs`.

## 9. Non-goals

- Full general-discrimination MLS2PLM (separate model-design PR per AGENTS.md).
- MH-RM engine (documented alternative for `K ≥ 3`).
- Polytomous responses; inner-product (HLSIRM-style) interaction term.

## 10. Phase 2 additions (QMC/MC-EM, scoring, FIPC — implemented)

Paper basis: Part II of the formula compilation (Wei & Tanner 1990; Booth &
Hobert 1999; Jank 2005; Meng & Schilling 1996; Bock & Mislevy 1982; Thissen,
Pommerich, Billeaud & Williams 1995; Lord & Wingersky 1984 via Cai 2015;
Kim & Cohen 1998; Hanson & Beguin 2002; Kim 2006; Sinharay & Haberman 2014).

- **Integration rules** (`nodes.rs`): the latent-space integral accepts
  tensor Gauss-Hermite (default, `K <= 3`), Halton QMC with an optional
  Cranley-Patterson shift (QMC-EM; `O(N^-1 (log N)^K)` error), or seeded
  Monte Carlo (MCEM). All deterministic given their parameters, so the
  Rust<->NumPy parity contract extends to them. `FitConfig(xi_rule=...,
  xi_points=..., xi_seed=...)`.
- **Scoring** (`scoring.rs`, all-Rust compute): EAP (Bock-Mislevy), MAP
  (damped posterior Newton, observed-information SEs), and EAPsum summed-
  score conversion tables via the Lord-Wingersky recursion run on the joint
  `(t, x)` node set. Priors are per-dimension `N(mean_d, sd_d^2)`: standard,
  group `(mu_g, sigma_g)`, cluster-conditional `N(u_hat_c, 1)`, or the
  multilevel marginal `N(0, sqrt(1 + sigma_u^2))` for unknown clusters
  (`serving_prior`). Serving exposes `method="eap"|"map"|"eapsum"` and the
  bundle embeds the conversion tables.
- **Fit statistics** (`fitstats.rs`, all-Rust compute): S-X2 with the
  `rms_residual` practical-significance effect size — added after the first
  31k-person run showed BH-significance alone removes 45/57 items (the
  chi-square is over-powered at large N); the screening MSQ gate uses infit
  only (outfit explodes under <1% pass rates); the `l_z*` screen threshold is
  configurable and its MAP `r_0` correction centers on the population prior
  mean (team intercepts / group means).
- **FIPC** (`Anchors` + `PopulationSpec::SingleFree`): anchored items (and
  optionally `tau`) frozen at supplied values, new items and the freed
  population `(mu_d, sigma_d)` estimated each EM cycle — the multiple-cycle
  prior-update variant (MWU-MEM-style) Kim (2006) found robust. PCA
  re-alignment is skipped so the anchor orientation is inherited.
  **Concurrent calibration** is the multigroup path plus structural
  missingness (Hanson-Beguin common-item design) — covered by
  `concurrent_calibration_two_forms_with_anchor_block`.
- **Compute placement**: every numeric path (estimation, scoring, fit
  statistics) executes in `mlsirm-core`; the Python layer is orchestration,
  I/O, and the NumPy parity references only.

## 11. Known numerical notes

- Multigroup/multilevel EM moves the quadrature nodes when `(mu, sigma)` /
  `sigma_u` update, so the quadrature APPROXIMATION of the marginal
  log-likelihood can dip by discretization error (~1e-4 on small fixtures)
  even though exact EM is monotone; tests allow 1e-3 absolute slack.
- The GPU E-step accumulates in f32 (~1e-4 relative noise): convergence
  tolerances below the noise floor never trigger — use a tolerance around
  `1e-5 * |loglik|` or an iteration budget for GPU runs. The M-step and the
  final EAP pass always run on the CPU in f64.
- 2PL-LSIRM slopes are weakly identified against item positions (the
  Bayesian original fixes `alpha_1 = 1`); the lognormal slope prior keeps
  them finite, and slope recovery needs materially more data than easiness
  recovery.
