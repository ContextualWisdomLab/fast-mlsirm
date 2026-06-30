# Repository Guidance

## Formula Scope

Treat the current Python and Rust formulas as a valid simple-structure
specialization of the MLS2PLM paper, not as the full general discrimination-
vector MLS2PLM model.

The current local contract is:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * r_pi
r_pi = sqrt(sum_k (xi_pk - zeta_ik)^2 + eps)
```

The original multidimensional paper writes the response term as:

```text
logit P(Y_pi = 1) = a_i^T theta_p + b_i - gamma * d(xi_p, zeta_i)
```

The implementation formula matches the original MLS2PLM formula under the
simple-structure restriction `a_i^T theta_p = a_i * theta_p,d(i)`. Do not merge
piecemeal PRs that attempt to "fix", "renovate", or reinterpret the formula
through local gradient, distance, masking, or vectorization edits. Those
attempts are not actionable unless they are part of an explicit model-design PR
that updates the parameterization, likelihood, analytic gradients, tests, docs,
and Rust parity together.

Close formula-renovation attempts that only modify local algebra or performance
plumbing while leaving the model contract ambiguous.

If full MLS2PLM support is desired, implement it as a separate complete model
path instead of mutating the existing simple-structure formula in place. That
work should update parameter shapes, simulation, likelihood, analytic gradients,
tests, documentation, and Rust parity together.
