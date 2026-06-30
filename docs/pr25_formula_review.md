# PR #25 Formula Review

Review target: https://github.com/ContextualWisdomLab/fast-mlsirm/pull/25

## Verdict

Do not merge PR #25 as a formula refactor. Close it as not actionable.

The attempted gradient/vectorization change is locally algebraic for the current
code path, but it should not be treated as a valid MLS2PLM formula renovation.
After checking the original MLSIRM/MLS2PLM papers, the repository's current
implementation is valid as a simple-structure specialization of the original
MLS2PLM model:

```text
eta_pi = exp(alpha_i) * theta_p,d(i) + b_i - exp(tau) * r_pi
```

The MLS2PLM paper's general multidimensional response model is instead:

```text
logit P(Y_pi = 1) = a_i^T theta_p + b_i - gamma * d(xi_p, zeta_i)
```

The implementation formula therefore matches the original MLS2PLM formula under
the simple-structure restriction `a_i^T theta_p = a_i * theta_p,d(i)`. That
restriction is a model choice, not an implementation detail that can be freely
rewritten into the full paper model through local gradient or distance
optimizations. Any future change that attempts to "fix" or "modernize" the
formula must first introduce an explicit model-design change, update parameter
shapes and constraints, and derive the full likelihood and gradient from the
paper.

## Paper Basis

Sources checked:

- Kang, I., & Jeon, M. (2025), "Multidimensional Latent Space Item
  Response Models: A Note on the Relativity of Conditional Dependence",
  Psychometrika 90(2), 799-826 (`doi:10.1017/psy.2025.5`).
- Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021),
  "Mapping Unobserved Item-Respondent Interactions: A Latent Space Item Response
  Model with Interaction Map" (`doi:10.1007/s11336-021-09762-5`).
- Local formula contract in `docs/prd_trd_summary.md`.

## Closure Recommendation

Close PR #25 and related Bolt formula/latent-distance optimization attempts
instead of merging them.

Rationale:

- PR #25 is conflicted with current `main`.
- It includes broad unrelated formatting churn and a committed `.coverage`
  artifact.
- It frames a local vectorization as a formula improvement while the real issue
  is model scope: current code already matches the original MLS2PLM formula as
  a simple-structure specialization, not as the full MLS2PLM discrimination-
  vector model.
- Keeping such PRs open encourages piecemeal formula edits without a full paper-
  aligned redesign.

Acceptable future work:

- A design PR that explicitly decides whether `fast-mlsirm` should remain a
  simple-structure MLS2PLM implementation or add a separate full MLS2PLM model.
- If adding full MLS2PLM, update parameterization, simulation, likelihood,
  analytic gradients, tests, documentation, and Rust parity together.
- Prefer implementing the full paper model as a separate, complete model path
  over mutating the existing simple-structure formula in place. The current
  formula is already a valid restricted model; replacing only one formula term
  would create a hybrid that is neither the current contract nor the full
  original paper model.
- Pure numerical optimizations may proceed only after proving they preserve the
  already-declared local formula contract and are not described as formula
  renovations.
