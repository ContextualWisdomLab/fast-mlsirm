# ADR-0021: Rust-owned residual interaction-map boundary

Status: **Accepted**
Date: 2026-08-25
Supersedes: none
Superseded by: none

## Context

Downstream measurement products need to inspect residual person-item
interactions after a fitted IRT main effect. Jeon et al. (2021) place those
interactions in a shared latent space; Gabriel (1971) defines the symmetric
biplot factorization used to display a residual matrix. Computing that map in
a product repository duplicates psychometric numerical policy and prevents
independent consumers from sharing one tested contract.

## Decision

fast-mlsirm owns `residual_interaction_map(observed, expected, axis_count)`.
The Rust core computes `R = Y - E`, excludes incomplete rows and columns
without zero filling, centers the admitted residual rectangle, performs the
Gabriel symmetric factorization, and returns coordinates, singular values,
axis inertia, truncated reconstruction `Rhat`, `U = R - Rhat`, and the exact
algebraic cross term `2 Rhat U / R^2`. The number of retained axes is required
from the caller; the library does not invent a display dimension.

The contract contains array indices only. Products retain responsibility for
domain identifiers, authorization, persistence, closest/farthest selection,
and UI wording. A consumer must not reproduce the factorization or cross-term
arithmetic locally.

## Consequences

- Numerical behavior is reusable, Rust-owned, and independently testable.
- Missing cells remain absent rather than becoming evidence-valued zeros.
- The cross term is an auditable algebraic identity, not a fitted weight,
  threshold, quality score, or heuristic.
- This API does not fit an LSIRM or replace the caller's GRM/GPCM fit.

## References

Gabriel, K. R. (1971). The biplot graphic display of matrices with application
to principal component analysis. *Biometrika, 58*(3), 453–467.
https://doi.org/10.1093/biomet/58.3.453

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping
unobserved item-respondent interactions: A latent space item response model
with interaction map. *Psychometrika, 86*(2), 378–403.
https://doi.org/10.1007/s11336-021-09762-5
