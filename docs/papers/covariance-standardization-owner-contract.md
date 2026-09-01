# Covariance standardization owner contract

## Scope

`mlsirm-core` owns the reusable static numerical map from a covariance matrix to its correlation matrix:

\[
R = D^{-1/2}\Sigma D^{-1/2},
\]

where `D` is the diagonal of strictly positive marginal variances. The scalar self-standardization is `(1 / sqrt(v)) * v * (1 / sqrt(v)) = 1` for finite `v > 0`.

This contract is domain-neutral. It does not decide event time, valid time, knowledge cutoff, state evolution, temporal identification, or product-specific model activation. Those policies belong to consuming bounded contexts such as TEPP Longitudinal Modeling.

## Triggering reuse case

TEPP PR #475 recovered the ctsem `TIPREDVARstd` scalar map inside `psychometric_core::event_time`. Review of that implementation showed that the EventTime admission rule is TEPP-owned, while the arithmetic itself is reusable covariance standardization and therefore belongs in fast-mlsirm. The owner implementation is tracked by issue #1720.

The TEPP evidence is preserved rather than copied as a product name here: Driver, Oud, and Voelkle describe the continuous-time model and the time-independent predictor covariance family; the 2017-era ctsem summary implementation forms standardized covariance quantities using inverse marginal standard deviations. fast-mlsirm exposes the generic normalization only. TEPP can later bind `TIPREDVARstd` through an ACL after a released version exists.

## TDD lineage

RED `738d3e5e7aa63b75f664fd17d9f422441bfa65e6` added an external crate contract requiring the versioned public module, scalar scale-invariance, a known matrix result, multiplicative scale invariance, and fail-closed malformed inputs before the production module existed. The subsequent implementation commits add the reusable Rust kernel and export it from the crate entrypoint.

## Numerical contract

`fast_mlsirm.covariance_standardization@1.0.0` requires finite covariance cells, a non-empty square row-major matrix, strictly positive diagonal variances, symmetry within an explicit binary64 tolerance, and pairwise covariance magnitude consistent with `|r| <= 1` up to floating-point tolerance.

The implementation divides sequentially by each marginal standard deviation rather than first multiplying two square roots. This avoids overflowing a representable correlation solely because `sqrt(v_i) * sqrt(v_j)` exceeds the finite binary64 range. Near-boundary correlations within rounding tolerance are clamped to `[-1, 1]`; materially invalid pairwise covariance fails closed.

The contract does not claim to prove full positive semidefiniteness. Model-specific PSD admission remains a separate invariant.

## Recovery and parity

The external contract covers positive scalar magnitudes from `f64::MIN_POSITIVE` through `f64::MAX`, malformed scalar inputs, a known 2x2 covariance/correlation pair, multiplicative scale invariance, shape errors, non-finite cells, non-positive diagonals, asymmetry, and pairwise covariance beyond the correlation bound.

Before TEPP removes its local duplicate, the TEPP adapter must prove parity with its preserved `TIPREDVARstd` fixtures while continuing to enforce EventTime semantics outside this kernel.

## Reference

Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous time structural equation modeling with R package ctsem. *Journal of Statistical Software, 77*(5), 1–35. https://doi.org/10.18637/jss.v077.i05
