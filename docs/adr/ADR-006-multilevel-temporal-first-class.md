# ADR-006: Multilevel, multiple-membership, and temporal structure are first-class

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

Psychometric and AI-evaluation data frequently contain respondents nested in organizations, items nested in testlets, observations associated with multiple groups, cross-classification, repeated occasions, rater assignment, task revisions, and changing model/rubric versions. Treating all response cells as exchangeable individual-level data risks atomistic fallacy, underestimated uncertainty, and invalid model comparison.

## Decision

When the data-generating design contains contextual or temporal structure, the core contracts must represent it explicitly and future estimators must either model it or fail closed rather than silently reduce it to an atomistic model.

Required contract concepts include:

- `context_dimension_id` distinct from `context_id`;
- nesting and cross-classification;
- weighted multiple membership with explicit weight policy;
- respondent/task/rater/occasion identities;
- exact task/rubric/model revision provenance;
- ordered temporal occasions;
- random-intercept/random-slope state specifications;
- discrete occasion-step state/autoregressive semantics where implemented.

A discrete occasion-step autoregressive coefficient is not a continuous-time coefficient. Interval-dependent or continuous-time transitions require a distinct parameterization, likelihood, documentation, and recovery study.

## Consequences

- Multilevel and temporal fields cannot be inferred from labels or collapsed silently.
- Estimators must diagnose disconnected/confounded designs where parameters are not identified.
- Recovery studies must reproduce the same clustering/membership/time design used by the feature.
- Cluster/block-aware likelihood or resampling units are required when iid assumptions would create pseudo-replication.

## Alternatives considered

1. **Flat observation table with caller-side correction** — rejected because it makes hierarchy/time invisible to validation, recovery, and model-comparison APIs.
2. **Treat every context as a generic latent-space interaction** — rejected because substantive hierarchy, testlet, rater, and temporal effects should be represented before residual interaction.
3. **Store timestamps but ignore semantics** — acceptable only as provenance; not acceptable as a claim of temporal modeling.

## References

Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel IRT model. *Psychometrika, 66*, 271–288. https://doi.org/10.1007/BF02294839

Uto, M. (2022). A Bayesian many-facet Rasch model with Markov modeling for rater severity drift. *Behavior Research Methods, 55*, 3910–3928.
