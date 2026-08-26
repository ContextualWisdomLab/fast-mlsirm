# ADR-0023: Rust-owned achieved finite-population proportion

Status: **Proposed**
Date: 2026-08-27

## Context

ADR 0022 designs a finite-population sample but deliberately does not turn an
observed sample into a population claim. Downstream coverage audits need a
terminal artifact that binds the exact design, rejects partial samples, and
reports both uncertainty and the achieved proportion. Treating `x / n` alone
as population coverage is invalid, especially when all sampled units succeed.

## Decision

`fast-mlsirm` owns `fast-mlsirm.achieved-proportion.v1` in Rust. Version 1
accepts one completed, one-stratum simple random sample without replacement
(SRSWOR). It binds the ADR 0022 design artifact SHA-256, `N`, its required
sample size `n`, the observed success count `x`, and the design confidence
level. A partial denominator, changed design, multiple strata, non-finite
confidence, or unsupported contract fails closed.

Rust reports the sample-proportion estimator

```text
p_hat = x / n
```

and the usual unbiased SRSWOR design-variance estimator

```text
Var_hat(p_hat) = (1 - n/N) p_hat (1 - p_hat) / (n - 1).
```

A census has variance zero. A non-census sample of size one is unavailable
because its sample variance cannot be estimated.

The two-sided interval is the exact equal-tailed hypergeometric interval
identified by Wang (2015) as the Konijn interval `C_O`, not Wang's subsequently
squeezed admissible Algorithm II interval. For
`X ~ Hyper(M, N, n)`, `q = (1 - confidence_level) / 2`, and `x > 0`,

```text
L_q(x) = max {m : Pr_(M=m-1)(X <= x-1) >= 1-q}
U_q(x) = N - L_q(n-x),
L_q(0) = 0.
```

The artifact returns the closed integer interval `[L, U]` for the unknown
population success count and the corresponding grid endpoints `[L/N, U/N]`.
It is exact in the coverage sense: minimum coverage over every `M` is at least
the declared confidence level. It is not described as globally shortest,
optimal, or admissible.

Rust evaluates hypergeometric log probabilities, uses adjacent-term recurrence
and log-sum-exp accumulation, then finds the monotone inversion boundary by
integer binary search. This avoids an `N`-sized allocation and keeps work
approximately `O(n log N)`. Python only validates, replays the design through
Rust, marshals the result, and rejects unknown versions.

## Acceptance evidence

- The interval reproduces Wang's published `C_O` table for `N=200`, `n=20`,
  and 95% confidence.
- Exhaustive enumeration over small finite populations verifies coverage for
  every true `M`, not by Monte Carlo.
- Tests verify monotone endpoints, success/failure complement symmetry, census
  collapse, extreme observations, input rejection, deterministic identities,
  and design-artifact binding.
- All-success samples retain a lower bound below one unless they are a census.

## Alternatives considered

- A normal/Wald interval was rejected because its minimum coverage can fall
  materially below nominal coverage and an all-success sample yields a
  degenerate interval.
- Wang's Algorithm II was not approximated: it requires the complete
  confidence-belt squeezing procedure. A future version may implement it under
  a distinct method and acceptance suite.
- Stratified and multistage terminal estimators remain unavailable until their
  estimand, variance, covariance, and interval contracts receive a separate
  decision.

## References

Cochran, W. G. (1977). *Sampling techniques* (3rd ed.). John Wiley & Sons.

Wang, W. (2015). Exact optimal confidence intervals for hypergeometric
parameters. *Journal of the American Statistical Association, 110*(512),
1491–1499. https://doi.org/10.1080/01621459.2014.966191
