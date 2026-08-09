# ADR-0004: Structural Model Selection, Context, and Time

- **Status:** accepted
- **Date:** 2026-08-09

## Context

The repository supports or is actively extending multiple measurement structures: unidimensional and multidimensional IRT, bifactor, higher-order, testlet, two-tier, many-facet, multilevel/multiple-membership, longitudinal/drift, and latent-space interaction. These structures are not interchangeable labels. Choosing the most flexible model by in-sample fit can make nuisance dependence look like substantive ability, make an omitted substantive factor look like latent distance, or turn rater effects into target quality.

## Decision

Use a staged structural-selection hierarchy:

1. define the score interpretation and observed design;
2. establish primary dimensionality with unidimensional/correlated/exploratory MIRT candidates as appropriate;
3. model known shared-stimulus/query dependence with testlet/secondary structure;
4. add rater/task/occasion facets when ratings create systematic measurement effects;
5. compare higher-order and bifactor representations only when a general-factor interpretation is substantively required;
6. use two-tier when multiple primary traits and identified secondary/testlet/method dimensions coexist;
7. preserve multilevel, cross-classified, weighted multiple-membership, and temporal/drift structure where scientifically relevant;
8. add latent-space interaction only for residual conditional dependence that remains after the interpretable factor/facet/testlet structure.

Choose the simplest identified model that remains competitive in held-out evidence and satisfies residual, scoreability, invariance/DIF, and recovery requirements.

## Model-relation rule

Model names do not decide the statistical comparison procedure. The implementation must determine or explicitly receive relation metadata derived from actual parameter constraints:

- regular nested;
- boundary nested;
- nonlinear-constraint nested;
- strictly non-nested;
- overlapping;
- indistinguishable;
- unknown.

Normal-theory non-nested selection is not allowed before a formal distinguishability result. Boundary cases such as zero variance or zero latent-space weight require boundary-aware inference, commonly parametric bootstrap LR or another validated procedure. Unknown relation fails closed.

## Bifactor / higher-order rule

A better bifactor in-sample fit is insufficient to justify general and specific scores. Bifactor scoreability requires indices/diagnostics whose assumptions hold, including stable general-factor coverage and relevant reliability/construct-replicability evidence. Higher-order and bifactor models can be related by proportionality constraints; therefore relation must be derived from parameterization rather than treated as universally non-nested.

## Testlet and local dependence rule

Shared passage/query/case dependence is a nuisance/local-dependence structure unless evidence supports a substantive trait interpretation. Residual Q3/LD-like evidence should be interpreted relative to the current factor/facet model.

## Multilevel and multiple-membership rule

Do not collapse organization/team/project/person or analogous hierarchies into one flat level when the scientific question depends on contextual effects. Membership dimensions are explicit and multiple memberships preserve declared weights. Estimators must reject disconnected or confounded designs that cannot identify requested effects.

## Temporal rule

Time must be represented deliberately. Ordered occasions and version drift are valid discrete structures; elapsed-time/continuous-time transitions require their own likelihood and parameter-recovery evidence. A timestamp field is not a continuous-time model.

## Recovery and selection evidence

Depending on the structure, use:

- relation-appropriate LR/Vuong/bootstrapped procedures;
- cluster-aware held-out marginal likelihood;
- residual-dependence reduction;
- DIF/invariance/fairness;
- score determinacy/reliability/scoreability;
- parameter and structure recovery (bias, RMSE, coverage, selection accuracy);
- bootstrap/split-sample stability;
- appropriate sign/permutation/rotation/Procrustes alignment.

## Rejected alternatives

1. **Always choose bifactor because fit is better.** Rejected due flexibility and scoreability risks.
2. **Always choose the model with the smallest BIC.** Rejected because the criterion can favor the wrong structural interpretation under realistic conditions.
3. **Treat latent space as a universal residual absorber.** Rejected because omitted factors/testlets/facets would become uninterpretable coordinates.
4. **Flatten hierarchy/time for convenience.** Rejected because it can create atomistic fallacy, biased uncertainty, and invalid contextual interpretation.

## References

- Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*, 581–612.
- Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*, 271–288.
- Kang, I., & Jeon, M. (2025). Multidimensional latent space item response models: A note on the relativity of conditional dependence. *Psychometrika, 90*, 799–826.
- Rijmen, F. (2010). Formal relations and an empirical comparison among the bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal of Educational Measurement, 47*, 361–372.
- Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods, 21*, 137–150.
- Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*, 664–684.
