# ADR-005: Model selection must be relation-safe and recovery-aware

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

The repository compares unidimensional, correlated multidimensional, bifactor, higher-order, testlet, two-tier, many-facet, response-family, and latent-space models. Their formal relationships vary with parameterization. Treating every pair as ordinary nested or ordinary non-nested can produce invalid likelihood-ratio or Vuong conclusions. Flexible models can also win in-sample fit while yielding unstable or uninterpretable scores.

## Decision

Factor retention and structural model selection are separate processes.

Before selecting between models, the comparison layer classifies the actual relationship as one of:

- regular nested;
- boundary/singular nested;
- nonlinear/constrained nested;
- strictly non-nested;
- overlapping;
- indistinguishable/degenerate;
- unknown.

The formal test must match the relation:

- regular LR/robust LR for regular nested cases;
- boundary-aware or parametric-bootstrap LR where null parameters lie on boundaries or directions are unidentified;
- formal Vuong distinguishability before non-nested selection;
- no winner when relation/distinguishability is unknown.

Model choice also considers held-out/cluster-aware predictive evidence, residual dependence, invariance/DIF, scoreability, parameter/structure recovery, and interpretability. When predictive evidence is practically equivalent, the simpler model is preferred if it satisfies the measurement purpose.

## Consequences

- APIs must represent indeterminate states explicitly instead of forcing `A` or `B`.
- A numeric likelihood-difference variance check is not labeled the formal Vuong distinguishability test.
- Bifactor fit does not authorize general/subscale score interpretation without scoreability evidence.
- Testlet/latent-space terms are introduced only after substantive/facet structure is represented.

## Alternatives considered

1. **AIC/BIC winner only** — rejected because information criteria do not settle formal relation, scoreability, or external generalization.
2. **Fit-index leaderboard** — rejected because more flexible structures can overfit and because score interpretation is a separate question.
3. **All model pairs use Vuong** — rejected because regular/boundary nested relations require different asymptotics or bootstrap procedures.

## References

Schneider, L., Chalmers, R. P., Debelak, R., & Merkle, E. C. (2020). Model selection of nested and non-nested item response models using Vuong tests. *Multivariate Behavioral Research, 55*(5), 664–684. https://doi.org/10.1080/00273171.2019.1664280

Rodriguez, A., Reise, S. P., & Haviland, M. G. (2016). Evaluating bifactor models: Calculating and interpreting statistical indices. *Psychological Methods, 21*(2), 137–150. https://doi.org/10.1037/met0000045
