# ADR-0005: Treat judges as fallible raters and fail closed in model selection

- Status: **Accepted**
- Date: 2026-08-09
- Owner: scoring/calibration/model-diagnostics layer

## Context

Human and automated evaluators differ in severity, criterion use, range, consistency, prompt/order sensitivity, and drift. Raw agreement or correlation can be high even when two raters are systematically offset or compressed. Separately, flexible psychometric models can improve in-sample fit without justifying the interpretation of every latent score.

## Decision

Human raters and LLM judges are measurement instruments represented as observations/facets, not ground truth by identity. Validation may use human anchors, known perturbations, authoritative evidence, or other external criteria, but the rater's label never bypasses measurement error by definition.

Structural model selection follows an evidence hierarchy:

1. identify the intended score/use and candidate model structures;
2. classify actual model relation/constraints where possible;
3. apply the relation-appropriate comparison (regular LR, boundary-aware bootstrap, or formal non-nested/overlap distinguishability/selection);
4. test held-out prediction with leakage-safe clusters;
5. inspect residual/local dependence, rater effects, DIF/invariance, and stability;
6. verify true-structure/parameter recovery under realistic simulation; and
7. separately test whether general/subscale scores are scoreable/interpretable.

If required relation or distinguishability evidence is missing, the API returns indeterminate rather than a fabricated winner.

## Invariants

1. `relation=unknown` cannot silently become `strictly_non_nested`.
2. Positive casewise log-likelihood variance is not named a formal Vuong distinguishability result.
3. Boundary parameters such as zero testlet/factor variance are not tested with an unjustified ordinary chi-square reference merely because one model appears nested.
4. Bifactor fit is distinct from bifactor scoreability; PUC/omega/ECV/H are reported only within their documented assumptions.
5. Latent-space terms are added after substantive dimensions/facets/testlets when residual interaction evidence justifies them; they do not replace omitted substantive dimensions.
6. Correlation is association evidence, not proof of absolute agreement, parameter recovery, fairness, or validity.
7. Rater disagreement can represent noise, severity, missing information, task ambiguity, or legitimate stakeholder perspective; the software preserves those distinctions where the design provides evidence.

## Alternatives considered

- **Average all judge scores:** rejected because item/rater/task effects are confounded.
- **Choose the model with the smallest AIC/BIC only:** rejected because relation, overfitting, local dependence, recovery, and score interpretation remain unresolved.
- **Always use the most complex bifactor/latent-space model:** rejected because flexibility can absorb misspecification and produce unstable/meaningless scores.

## Failure and recovery

When a fit is nonidentified, non-finite, relation-unknown, distinguishability-unknown, locally dependent beyond the model, or not scoreable, the result is diagnostic/indeterminate. Recovery means changing the design/model or collecting evidence; it does not mean relaxing the gate until a preferred result appears.

## Verification

- Many-facet connectedness and rater-effect tests.
- agreement beyond correlation where applicable.
- relation-safe comparison tests.
- cluster-aware held-out tests.
- DIF/invariance/local-dependence diagnostics.
- true-parameter/model-recovery simulations.
- scoreability diagnostics with explicit applicability rules.

## Scientific basis

This decision is grounded in many-facet measurement, IRT model-comparison, bifactor scoreability, latent-space conditional-dependence, and automated-scoring validity literature documented in `AGENTS.md`, feature doctoring, and `docs/doctoring/conversation_architecture_baseline.md`.

## Consequences

The library may return “not established” more often than simpler evaluation wrappers. That is intentional: the product's value is defensible measurement evidence rather than forced rankings.
