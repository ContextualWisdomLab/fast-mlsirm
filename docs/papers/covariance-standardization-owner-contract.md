# Covariance standardization owner contract

## Scope

`mlsirm-core` owns the reusable static numerical map from a covariance matrix to its correlation matrix:

\[
R = D^{-1/2}\Sigma D^{-1/2},
\]

where `D` is the diagonal of strictly positive marginal variances. The scalar self-standardization is `(1 / sqrt(v)) * v * (1 / sqrt(v)) = 1` for finite `v > 0`. After validating the scalar variance, the implementation evaluates the algebraically equivalent ratio `v / v`, so every admitted finite positive binary64 value—including the smallest positive subnormal and `f64::MAX`—returns the exact binary64 representation of `1.0` rather than accumulating avoidable square-root rounding.

This contract is intentionally domain-neutral. It does not decide event time, valid time, knowledge cutoff, state evolution, temporal identification, or product-specific model activation. Those policies belong to consuming bounded contexts such as TEPP Longitudinal Modeling.

## Triggering reuse case

TEPP PR #475 recovered the ctsem `TIPREDVARstd` scalar map inside `psychometric_core::event_time`. Review of that implementation showed that the EventTime admission rule is TEPP-owned, while the arithmetic itself is reusable covariance standardization and therefore belongs in fast-mlsirm. The owner implementation is tracked by issue #1720.

The TEPP evidence is preserved rather than copied as a product name here: Driver, Oud, and Voelkle describe the continuous-time model and the time-independent predictor covariance family; the 2017-era ctsem summary implementation forms standardized covariance quantities using inverse marginal standard deviations. fast-mlsirm exposes the generic normalization only. TEPP can later bind `TIPREDVARstd` through an ACL after a released version exists.

## Numerical contract

`fast_mlsirm.covariance_standardization@1.0.0` requires:

- finite covariance cells;
- a non-empty square row-major matrix;
- strictly positive diagonal variances;
- mirrored covariance cells that are exactly equal in the supplied binary64 representation;
- exact pairwise admissibility of the represented values under `c² <= v_i * v_j`.

Pairwise admission does not use an empirical epsilon. Each finite binary64 value is decomposed into its exact integer significand and power-of-two exponent, and the two products in `c² <= v_i * v_j` are compared exactly with `u128` significand products plus exponent alignment. Consequently, a genuinely invalid represented covariance fails closed even if later floating-point division would round its correlation back into range.

The correlation itself is evaluated without first multiplying the two marginal standard deviations. After exact pairwise admission, the covariance is divided by the smaller standard deviation first and the larger standard deviation second. For an admissible pair the first quotient is bounded by the larger standard deviation, which avoids overflow and also prevents an avoidable zero caused by dividing a very small covariance by the larger scale first. This ordering makes the numerical result invariant to exchanging the two variables even when their variances differ by hundreds of orders of magnitude.

The final division can still round a mathematically admissible boundary correlation one representable value outside `[-1, 1]`. After the exact admissibility proof, and only after that proof, the computed value is projected back to `[-1, 1]`. This projection is therefore a consequence of the exact represented-input bound rather than a tolerance for invalid covariance.

The contract does not claim to prove full positive semidefiniteness. Model-specific PSD admission remains a separate invariant. Callers that wish to accept approximately symmetric observations must define and validate that preprocessing policy before calling this exact low-level kernel.

## Recovery and parity

Unit and public-contract fixtures cover the smallest positive subnormal scalar variance, `f64::MIN_POSITIVE`, ordinary values including `3.0`, very large finite values through `f64::MAX`, exact binary64 unit recovery, malformed scalar inputs, a known 2x2 covariance/correlation pair, multiplicative scale invariance, shape errors, non-finite cells, non-positive diagonals, exact-symmetry rejection, pairwise covariance beyond the exact correlation bound, zero covariance with a subnormal variance, a finite exact-valid boundary case whose sequential binary64 divisions round to `next_up(1.0)` before the bound-certified projection, and permutation invariance for an extreme-scale covariance whose correlation would underflow to zero if the larger marginal standard deviation were divided first.

Before TEPP removes its local duplicate, the TEPP adapter must prove parity with its preserved `TIPREDVARstd` fixtures while continuing to enforce EventTime semantics outside this kernel.

## Reference

Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous time structural equation modeling with R package ctsem. *Journal of Statistical Software, 77*(5), 1–35. https://doi.org/10.18637/jss.v077.i05
