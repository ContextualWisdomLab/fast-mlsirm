# ADR-0004 — Relation-Safe Model Selection and Scoreability

- **Status:** Accepted
- **Date:** 2026-08-09
- **Decision owner:** `fast-mlsirm`
- **Implementation status:** partially implemented; unsupported relation classes remain fail-closed requirements

## Context

Unidimensional, correlated MIRT, bifactor, higher-order, testlet, two-tier, many-facet and latent-space models can be nested, boundary-nested, constrained, strictly non-nested, overlapping or practically indistinguishable depending on actual parameterization. Choosing a statistical test from model names alone can produce invalid winners. Separately, a model can fit well while its proposed scores are poorly determined or uninterpretable.

## Decision

Model selection follows this order:

1. identify the intended score interpretation and design;
2. classify the actual model relationship/regularity;
3. apply relation-appropriate inference;
4. compare cluster-aware held-out predictive behavior;
5. inspect residual/local dependence, DIF/invariance and convergence;
6. perform true-structure/parameter recovery;
7. separately evaluate scoreability/reliability/determinacy;
8. prefer the simplest scientifically adequate model.

Unknown relation, missing distinguishability evidence, boundary uncertainty or practical indistinguishability produces an indeterminate decision rather than a forced model winner.

## Relation rules

- regular nested constraints → LR/scaled LR when regularity conditions hold;
- boundary/singular/null-unidentified parameters → parametric bootstrap or other justified boundary-aware procedure;
- strictly non-nested/overlapping models → formal distinguishability before Vuong-style selection;
- unknown/indistinguishable → no preferred model.

Casewise likelihood units respect the independent sampling/design cluster. Response cells from the same person/query/testlet/rater are not automatically independent observations.

## Bifactor-specific rule

Bifactor fit is not authorization to publish a general or specific score. Applicability of the general factor, ECV/item-ECV, PUC assumptions, omega-H/omega-HS or other reliability/determinacy evidence and score recovery remain separate. Latent-response omega obtained through a logistic residual-variance mapping is labelled as latent-response evidence and is not categorical observed-score omega.

## Alternatives considered

1. Lowest AIC/BIC only — rejected as insufficient across flexible/nonregular structures.
2. Highest CFI/lowest RMSEA only — rejected; in-sample fit is not scoreability or prediction.
3. Always choose bifactor/latent-space as most flexible — rejected.
4. Relation-aware multi-evidence selection — accepted.

## Consequences

Some comparisons intentionally return no winner. The API must expose enough relation/test/evidence metadata to make that defensible. Additional computation such as cross-validation, bootstrap and recovery is accepted because correctness matters more than raw selection speed.

## Failure / degraded behavior

If score vectors/information required for a formal distinguishability test are unavailable, the system reports that evidence gap and may still return descriptive likelihood/AIC/BIC diagnostics without promoting them to a formal preference.

## Verification

- true-structure simulation confusion matrices;
- false-selection rates and parameter bias/RMSE/coverage;
- boundary/bootstrap oracle cases;
- degenerate/identical casewise likelihoods;
- clustered held-out validation;
- bifactor scoreability edge cases;
- stable typed statuses rather than brittle exception-string parsing.

## Sources

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*(5), 664–684.

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods, 21*(2), 137–150.

Rijmen, F. (2010). Formal relations and an empirical comparison among the bi-factor, the testlet, and a second-order multidimensional IRT model. *Journal of Educational Measurement, 47*(3), 361–372.

Cai, L. (2010). A two-tier full-information item factor analysis model with applications. *Psychometrika, 75*, 581–612.

## Supersession criteria

Supersede when a new comparison framework provides stronger formally verified coverage of supported model relations while retaining fail-closed behavior and recovery/predictive evidence.
