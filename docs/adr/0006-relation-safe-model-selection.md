# ADR-0006: Relation-safe factor and measurement-model selection

Status: **Accepted**  
Date: 2026-08-09

## Context

The product supports or is extending multiple model structures: unidimensional and correlated MIRT, bifactor, higher-order, testlet, two-tier, multifaceted and latent-space models. These structures are not reliably classifiable as nested or non-nested by their names. Additional factor/testlet/latent-space effects may also create boundary or singular null hypotheses where ordinary chi-square LR theory is invalid.

Selecting the model with the largest in-sample likelihood, lowest BIC, or most attractive plot can overfit and can create unsupported score interpretations.

## Decision

Factor retention and structural model selection are separate decisions.

### Stage 1: substantive factor-retention candidates

Use data-type-appropriate retention evidence such as exploratory MIRT/EFA, parallel/MAP/network methods where appropriate, fit/residual diagnostics, theory and cross-validation to define a small candidate factor-count set.

### Stage 2: structural relation classification

Classify each pair from actual parameter constraints:

- regular nested;
- boundary/singular nested;
- nonlinear-constraint nested;
- strictly non-nested;
- overlapping/indistinguishable;
- unknown.

Higher-order versus bifactor, testlet versus bifactor, or a latent-space extension must not be hard-coded as non-nested solely from model labels.

### Stage 3: relation-appropriate inferential evidence

- regular nested -> appropriate likelihood-ratio/robust equivalent;
- boundary/singular -> boundary-aware or parametric-bootstrap LR;
- strictly non-nested/overlapping -> formal Vuong distinguishability before normal-theory selection;
- unknown -> no model preference until relation is established.

A numerical positive variance of casewise log-likelihood differences is not the full formal Vuong distinguishability test.

### Stage 4: operational predictive evidence

Use cluster-aware held-out likelihood at the level relevant to deployment, such as query/testlet, respondent/system, rater family or domain. Random response-cell splitting is avoided when it leaks the same person/query/rater into train and validation.

### Stage 5: interpretation evidence

Inspect residual dependence, DIF/invariance, factor determinacy/score reliability, bifactor scoreability, rotation/stability and external validity as appropriate.

### Stage 6: recovery

Simulate realistic generating structures and evaluate model-selection accuracy plus parameter bias/RMSE/coverage/convergence.

### Selection rule

Prefer the simplest model whose predictive performance is practically competitive and whose identification, residual, invariance, scoreability and recovery conditions support the intended interpretation.

## Consequences

The system may return `indeterminate`, `requires_distinguishability_test`, or `requires_likelihood_ratio` rather than a winner. This is intended product safety, not missing functionality.

## Alternatives considered

- **AIC/BIC-only selection.** Rejected as insufficient for flexible and boundary models.
- **Always select bifactor when it fits better.** Rejected because bifactor flexibility does not establish scoreability.
- **Always add latent space for residual fit.** Rejected; latent space is residual interaction after substantive/facet/testlet structure and must improve held-out/recovery evidence.

## Research and standards basis

This ADR is about factor retention and structural model choice, including when a latent-space residual interaction may be added. Score interpretation after a selected model remains governed by AERA, APA, and NCME (2014). NIST, OWASP, and CWE catalogs are not the methodological basis.

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*, 581–612. https://doi.org/10.1007/s11336-010-9178-0

Jeon, M., Jin, I. H., Schweinberger, M., & Baugh, S. (2021). Mapping unobserved item-respondent interactions: A latent space item response model with interaction map. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5

Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*(2), 799–826. https://doi.org/10.1017/psy.2025.5

Preacher, K. J., Zhang, G., Kim, C., & Mels, G. (2013). Choosing the optimal number of factors in exploratory factor analysis: A model selection perspective. *Multivariate Behavioral Research, 48*, 28–56. https://doi.org/10.1080/00273171.2012.710386

Rijmen, F. (2010). Formal relations and an empirical comparison among the bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal of Educational Measurement, 47*, 361–372. https://doi.org/10.1111/j.1745-3984.2010.00118.x

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*, 664–684. https://doi.org/10.1080/00273171.2019.1664280
