# ADR-0020: Joint MAP hierarchical continuous-time AR(1) Rasch slice

Status: **Proposed**
Date: 2026-08-17
Supersedes: none
Superseded by: none

## Context

ADR-0019 on the live longitudinal state-engine branch (`#976`) owns an
independent per-respondent OLS trend and a caller-supplied discrete AR
predictor. Those kernels are scientifically valid for their stated estimands
and must not be relabeled as population random effects, estimated
autoregression, interval coverage, or continuous-time transitions.

The next product gap is the smallest jointly estimated longitudinal
latent-state IRT slice that *does* estimate shared population hyperparameters,
uses elapsed time in the transition, and reports uncertainty. Open PRs `#978`,
`#979`, and `#981` do not implement that gap. Closed `#848` is the same OLS/AR
tree as `#976`. Combining estimated multiple-membership context effects
`u_h` with person-occasion states is a larger identification problem and is
not this slice.

Journal PDFs for the governing methods are copyrighted. This record cites,
links, and summarizes them rather than attaching the PDFs.

## Decision drivers

- Honest estimand labels: do not call OLS/AR random effects if they are not.
- Hierarchical pooling requires shared population parameters and shrinkage.
- Irregular occasion spacing requires a continuous-time or explicitly
  time-scaled transition, not a discrete `phi` reused across unequal gaps.
- Uncertainty must be computed and interval coverage tested against known
  states, without inventing 95% hyperparameter coverage from five seeds.
- Multiple-membership / crossed grouping is compatible only if the joint
  likelihood actually estimates those effects; otherwise exclude it in
  contract language.
- GPU parity is honest only when an existing GPU abstraction implements the
  same estimand.

## Ownership and dependency direction

ADR-0028 governs temporal/event composition for this Proposed CT-AR Rasch numerical kernel.
TEPP owns event ontology, temporal validity, event ordering, changing-membership history, longitudinal leakage policy, and temporal/event composition.
fast-mlsirm owns the joint MAP likelihood, elapsed-time transition arithmetic, optimization, and uncertainty calculations over explicit supplied occasion/time carriers.
TEPP-originated carrier designs enter only through the versioned, immutable Anti-Corruption Layer defined by ADR-0028; this ADR does not authorize cross-service SQL, direct TEPP database access, or a hidden TEPP runtime dependency.

`ContextualWisdomLab/fast-mlsirm` owns this reusable measurement kernel.
Psychometrics Commons and other hosted products consume the sealed design and
returned diagnostics. This slice depends on the `#976` longitudinal design
handoff and must not recreate hosted session, consent, persistence, or HTTP
runtime. `kaefa` and `aFIPC` are not oracles.

## Decision

Add a separate Rust-owned joint MAP estimator behind a new Python entry point
`fit_hierarchical_longitudinal_irt`:

```text
logit P(Y_pti = 1) = theta_pt - b_i,   sum_i b_i = 0
theta_p,1 ~ N(mu, tau^2)
theta_p,t | theta_p,t-1 ~ N(
    mu + exp(-lambda * Delta_pt) * (theta_p,t-1 - mu),
    tau^2 * (1 - exp(-2 * lambda * Delta_pt))
)
```

`Delta_pt` is elapsed time in days from exact millisecond offsets. Packed
parameters are `[mu, log tau, log lambda, b, theta]`. Optimization is the
existing Rust L-BFGS path. Person shards run on scoped CPU threads and reduce
in respondent order.

Normative metadata:

- `estimand_scope = "joint_map_hierarchical_ctar_rasch"`
- `transition_kind = "continuous_time_ar1_ou"`
- `interval_kind = "wald_conditional_hyperparameter_observed_information"`
- `engine = "rust_cpu_multithreaded"`
- `population_random_effects_estimated = True`
- `ar_coefficient_estimated = True`
- `ar_coefficient_source = "joint_map"`
- `multiple_membership_estimated = False`
- `gpu_parity = False`

State standard errors use the person-block measurement observed information
only. The hierarchical prior regularizes the joint MAP point estimates; it is
not treated as known truth when forming Wald state intervals.
Hyperparameter standard errors use a 3×3 finite-difference Hessian on
`(mu, log tau, log lambda)` plus the delta method for `tau` and `lambda`, while
holding fitted item intercepts and latent states fixed. They are conditional
observed-information Wald 95% intervals, not profile or marginal intervals.
Identification flags are returned rather than invented when the Hessian is not
usable.

The `#976` OLS/AR entry point and its metadata remain unchanged.

## Invariants / acceptance evidence

1. Worker counts do not change the joint MAP state vector.
2. Multi-seed recovery against known generating states and transition
   parameters reports RMSE/bias and state-interval coverage inside
   documented MAP bounds. Those bounds are not a claim of unbiased ML or
   exact 95% hyperparameter coverage.
3. Missing responses are excluded; irregular gaps change `exp(-lambda Delta)`.
4. Invalid designs, non-binary responses, and invalid optimizer controls fail
   closed with package-owned errors.
5. Metadata never uses OLS or caller-supplied-AR estimand labels.

## Non-goals and claims not made

- Not independent respondent OLS and not caller-supplied discrete AR.
- Not Fox and Glas (2001) Gibbs sampling.
- Not Jeon and Rabe-Hesketh (2016) adaptive-quadrature ML.
- Not estimated multiple-membership or crossed `u_h` in this joint likelihood.
- Not GPU parity. The existing wgpu path owns MLSIRM distance/likelihood
  kernels in f32, a different estimand.
- Not a claim that five recovery seeds establish frequentist 95% coverage
  for `(mu, tau, lambda)`.

## Consequences and trade-offs

### Benefits

- Smallest scientifically valid joint longitudinal IRT slice on top of `#976`.
- Elapsed-time transitions and hierarchical shrinkage are named honestly.
- Python remains marshalling-only.

### Costs / risks

- Joint MAP shrinks `tau` and person states toward `mu`.
- Conditional Wald observed-information intervals are local and can be
  unidentified; profile intervals would require re-optimizing item and state
  nuisance blocks for every hyperparameter perturbation.
- Crossed / multiple-membership structure still requires a later joint model.

## Alternatives considered

### Relabel the ADR-0019 OLS/AR kernels as random effects

Rejected. Those kernels do not estimate a population distribution or `phi`.

### Discrete time-scaled AR with caller `phi`

Rejected as the next slice. It would still leave the coefficient
caller-supplied.

### Full MMMC + longitudinal joint likelihood

Deferred. Estimating context-level `u_h` together with CT-AR person states
needs a separate identification and recovery design (Fox & Glas; Browne,
Goldstein, & Rasbash).

### GPU kernel for this objective

Rejected. The current GPU abstraction does not implement this estimand.

## Failure, degraded, and recovery behavior

Invalid designs, non-finite packed parameters, degenerate OU transitions, and
worker join failures return package-owned errors. Unidentified Hessians set
interval flags to false and leave the corresponding bounds as NaN. Rollback
is removal of the new module and entry point; the `#976` OLS/AR layer remains.

## Security and privacy implications

No new credentials, network access, or raw source retention. Response
matrices are bounded before native allocation. Hostile Python mappings and
Boolean-as-number coercion fail closed.

## Compatibility, migration, and rollback

New public names only. Existing `fit_longitudinal_state` metadata is
unchanged. The sealed `LongitudinalDesign` is reused for occasion identity
and is not reinterpreted as this IRT estimand. Stacked on `#976`; do not
merge to `main` without that dependency.

## Verification and release evidence

- Rust unit tests: OU helpers, analytic vs finite-difference gradients,
  fail-closed validation, worker determinism, unused-shard join errors,
  missing/irregular time, tridiagonal/Hessian failures, multi-seed recovery.
- Python tests: public marshalling, recovery, metadata honesty, binding
  bounds, and simulator controls.
- Cargo, pytest, package, existing fuzz, and existing GPU-smoke evidence.
  This slice adds no GPU kernel, so GPU-smoke remains the MLSIRM path.

## Research and standards basis

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839

Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for
longitudinal item analysis. *Psychometrika, 81*(3), 830–850.
https://doi.org/10.1007/s11336-015-9489-2

Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal
data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876

Oravecz, Z., Tuerlinckx, F., & Vandekerckhove, J. (2011). A hierarchical
latent stochastic differential equation model for affective dynamics.
*Psychological Methods, 16*(2), 468–490. https://doi.org/10.1037/a0024375

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
103–124. https://doi.org/10.1177/1471082X0100100202

## Follow-ups

1. Jointly estimate context-level `u_h` with longitudinal states (Fox & Glas
   plus Browne et al. MMMC), after a dedicated identification study.
2. Adaptive-quadrature or Gibbs alternatives if MAP shrinkage is insufficient
   for a stated inferential target.
3. GPU only if a kernel implements this exact objective and passes CPU
   parity.

## Reversal / supersession conditions

A later ADR should supersede this record if the joint likelihood changes
(full discrimination-vector MLS2PLM, estimated MMMC, or a different
transition family), if recovery evidence falsifies the stated MAP bounds, or
if a GPU path is added for this estimand.
