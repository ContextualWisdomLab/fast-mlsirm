# ADR-0026: Public polytomous prediction boundary

Status: **Proposed**
Date: 2026-08-23

## Context

Downstream consumers need GRM/GPCM category probabilities and expected category
scores from a fitted item bank. The package already owns the stable Samejima
(1969) and Muraki (1992) category kernels in Rust, but only private `_core`
cell helpers expose them. Consumers therefore risk reimplementing parameter
conversion and probability arithmetic.

## Decision

Expose `polytomous_category_probabilities(fit, theta)` and
`polytomous_expected_response(fit, theta)` from the Python package. Both accept
the existing `PolytomousFit` and a finite one-dimensional trait vector. A
single Rust batch kernel validates item parameters, computes probabilities,
and derives expected integer category scores. PyO3 remains an implementation
boundary; consumers never import `_core`.

The output shapes are persons x items x categories and persons x items,
respectively. GRM uses the fitted decreasing cumulative-boundary intercepts;
GPCM uses the fitted additive category intercepts. No alternate threshold or
step convention is inferred.

## Evidence and consequences

Tests cover GRM/GPCM normalization, expected-score identity, extreme finite
traits, invalid parameters, and delegation through the compiled Rust module.
This adds no dependency and no second numerical implementation. Samejima
(1969) grounds GRM cumulative probabilities; Muraki (1992, 1993) grounds GPCM
category probabilities and expected integer scores.

## References

- Samejima, F. (1969). *Estimation of latent ability using a response pattern
  of graded scores*. Psychometrika, 34(S1), 1-97.
  https://doi.org/10.1007/BF03372160
- Muraki, E. (1992). A generalized partial credit model: Application of an EM
  algorithm. *Applied Psychological Measurement, 16*(2), 159-176.
  https://doi.org/10.1177/014662169201600206
- Muraki, E. (1993). Information functions of the generalized partial credit
  model. *Applied Psychological Measurement, 17*(4), 351-363.
  https://doi.org/10.1177/014662169301700403
