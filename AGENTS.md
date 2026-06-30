# Repository Guidance

## Paper-First Research

Before changing model formulas, fit diagnostics, estimators, simulation
contracts, or interpretation-facing outputs, research the relevant MLSIRM,
MLS2PLM, and psychometric fit-statistic literature first. Summarize the paper
basis in the plan before editing code, and keep the implementation inside the
paper-supported scope unless the task explicitly asks for a new model-design
PR.

## Key Articles

Start with these references before model or fit-diagnostic work:

- Kang, I., & Jeon, M. (2025). "Multidimensional Latent Space Item
  Response Models: A Note on the Relativity of Conditional Dependence."
  Psychometrika, 90(2), 799-826. doi:10.1017/psy.2025.5.
- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). "Mapping
  Unobserved Item-Respondent Interactions: A Latent Space Item Response
  Model with Interaction Map." Psychometrika, 86(2), 378-403.
  doi:10.1007/s11336-021-09762-5.
- Molenaar, D., & Jeon, M. (2026). "Regularized Joint Maximum Likelihood
  Estimation of Latent Space Item Response Models." Psychometrika, 91,
  335-359. doi:10.1017/psy.2025.10068.
- Orlando, M., & Thissen, D. (2000). "Likelihood-Based Item-Fit Indices
  for Dichotomous Item Response Theory Models." Applied Psychological
  Measurement, 24, 50-64.
- Drasgow, F., Levine, M. V., & Williams, E. A. (1985). "Appropriateness
  Measurement with Polychotomous Item Response Models and Standardized
  Indices." British Journal of Mathematical and Statistical Psychology,
  38(1), 67-86. doi:10.1111/j.2044-8317.1985.tb00817.x.

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
