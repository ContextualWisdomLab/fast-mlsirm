# ADR-0017: Rust-owned longitudinal state layer

Status: **Proposed**
Date: 2026-08-14

## Context

The sealed `fast_mlsirm.multilevel` contracts preserve respondent identity,
sequence, exact time offsets, revision provenance, weighted contexts, and
missing occasions. A contract without a numerical consumer still leaves a
product integration gap: a report or evaluation workflow cannot distinguish a
stable respondent level from a time-varying state. At the same time, a full
multilevel IRT estimator would introduce a much larger identification,
uncertainty, GPU, and recovery surface.

## Decision

Add a narrow Rust-owned state layer behind the existing Python contract and
PyO3 boundary:

1. `random_intercept_slope` is retained as the state-specification wire label,
   but the fitted estimand is an **independent per-respondent OLS trend**. Each
   respondent's finite observations are fitted by ordinary least squares on
   exact millisecond offsets converted to days. With
   `x_pi = (t_pi - t_p1) / 86,400,000`, the state is
   \(\hat\eta_{pi}=\hat\alpha_p+\hat\beta_p x_{pi}\). A respondent with fewer
   than two distinct observed times receives a zero slope rather than an
   invented trend. This path does not estimate a population random-effects
   distribution and applies no multilevel shrinkage.
2. `stationary_autoregressive` is likewise a state-specification wire label for
   a **discrete AR state predictor**. It uses the sealed caller-supplied
   `-1 < phi < 1` value and actual `sequence_index` gaps. Missing observations
   remain in the output state and do not reset the last observed state. The
   coefficient is a discrete-occasion parameter; this layer does not estimate
   \(\phi\) or its uncertainty. Exact calendar offsets are retained for audit
   and are not silently converted into continuous-time decay.
3. Respondents are independently sharded over scoped Rust CPU threads. Results
   are reduced in respondent order, making worker-count changes deterministic.
   Worker join failures are converted to the stable package-owned Rust error
   `longitudinal worker failed` rather than unwinding across the PyO3 boundary.
4. PyO3 performs only bounded array marshalling. Python exposes the returned
   state, intercepts, slopes, RMSE, counts, engine identity, and aligned
   respondent/occasion identifiers.

The Python result metadata is normative for interpretation. The OLS path emits
`estimand_scope="independent_respondent_ols_trend"`,
`population_random_effects_estimated=False`, and
`ar_coefficient_source="not_applicable"`. The AR path emits
`estimand_scope="discrete_ar_state_prediction"`,
`population_random_effects_estimated=False`, `ar_coefficient_estimated=False`,
and `ar_coefficient_source="caller_supplied"`. Consumers must use these fields
rather than infer an estimand from the compatibility wire label alone.

The weighted multiple-membership contextual predictor remains a separate Rust
kernel. This ADR does not claim Bayesian random-effect integration, joint item
and context likelihood estimation, interval uncertainty, continuous-time
transitions, or GPU parity for the recurrent state recurrence. Those require
their own equations, identification rules, and recovery evidence.

## Consequences

LineageWeave and other consumers can call one provider-neutral state estimator
without copying arithmetic into Python or treating missing occasions as absent
rows. A single-period report group can be represented honestly as a state
observation, but it cannot be reported as evidence of temporal trend. Consumers
must persist the state specification, design fingerprint, engine version, and
RMSE/count diagnostics before interpreting a score.

The current state layer is intentionally not a replacement for fast-mlsirm's
full psychometric calibration path. Production promotion remains gated on
protected integration, true-parameter recovery across nested/crossed/weighted
membership and unbalanced temporal designs, interval coverage, and GPU/CPU
parity where a GPU state implementation is justified.

## Verification

- Rust unit tests recover two respondent slopes exactly and compare one and
  many worker counts.
- Rust/Python tests preserve a missing occasion and reject malformed offsets,
  state kinds, coefficients, and worker counts.
- Rust regression coverage preserves valid slopes at millisecond-scale time
  intervals rather than treating a small centered sum of squares as a missing
  trend.
- Python tests use a non-contiguous sequence gap and verify that AR prediction
  uses the declared discrete gap rather than array position or calendar days.
- The PyO3 extension is built from the root Maturin project and exposes
  `fit_longitudinal_state` through the existing multilevel loader.

## Research basis (APA 7)

Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
multiple classification (MMMC) models. *Statistical Modelling, 1*(2), 103–124.
https://doi.org/10.1177/1471082X0100100202

Embretson, S. E. (1991). A multidimensional latent trait model for measuring
learning and change. *Psychometrika, 56*, 495–515.
https://doi.org/10.1007/BF02294487

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT
model using Gibbs sampling. *Psychometrika, 66*, 271–288.
https://doi.org/10.1007/BF02294839

Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for
longitudinal item analysis. *Psychometrika, 81*(3), 830–850.
https://doi.org/10.1007/s11336-015-9489-2

Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal
data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876
