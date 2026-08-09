# fast-mlsirm Architecture Decision Records

This directory is the authoritative decision log for architecture, scientific interpretation, trust boundaries, and cross-repository ownership decisions that materially affect `fast-mlsirm`.

## Status vocabulary

- **Accepted** — implemented or governing current protected-main behavior/policy.
- **Proposed** — desired design that is not yet fully implemented or protected-integrated.
- **Deprecated** — retained for history but no longer governs new work.
- **Superseded** — replaced by a named later ADR.

A conversation, issue, PR body, design note, or paper summary is not an Accepted decision by itself. Accepted ADRs must match current code/policy or explicitly describe an accepted invariant whose implementation is tracked.

## Decision index

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-domain-neutral-measurement-boundary.md) | Accepted | `fast-mlsirm` owns reusable measurement/psychometric contracts and kernels; hosted runtime belongs downstream. |
| [0002](0002-rust-first-numerical-ownership.md) | Accepted | Rust owns production psychometric arithmetic; Python validates/orchestrates/reports and retains governed reference/fallback paths. |
| [0003](0003-content-addressed-measurement-contracts.md) | Accepted | Assessment/rubric/scoring artifacts use canonical versioned, content-addressed provenance and replay verification. |
| [0004](0004-governed-rubric-item-bank-lifecycle.md) | Proposed | Build candidate-blind evidence-grounded rubric/item generation into a governed psychometric item-bank lifecycle. |
| [0005](0005-automated-scoring-raters.md) | Accepted | Human and automated scorers are fallible raters; calibration/validation must model rater effects and preserve terminal states. |
| [0006](0006-relation-safe-model-selection.md) | Accepted | Factor retention and structural model choice are distinct; model comparison is relation-safe and fail-closed when distinguishability is unknown. |
| [0007](0007-multilevel-multiple-membership-temporal.md) | Proposed | Multilevel, cross-classified, multiple-membership and temporal structure are first-class; Rust estimators require recovery evidence before production release. |
| [0008](0008-true-parameter-recovery-ci.md) | Accepted | True-parameter recovery/coverage, not correlation alone, is the core scientific CI evidence for numerical estimators. |
| [0009](0009-adaptive-rotation-selection.md) | Proposed | Rotation uses an extensible criterion registry, deterministic multi-start and criterion-neutral empirical selection; no universal best criterion. |
| [0010](0010-llm-orchestration-and-credentials.md) | Accepted | Model-backed automation uses provider-neutral boundaries, NVIDIA NIM credentials where needed, and never uses Copilot credentials for development scheduling. |

## ADR completeness rule

A material decision should have an ADR when it changes one or more of:

- repository/bounded-context ownership;
- public serialized contract or versioning rule;
- psychometric model parameterization/identification/interpretation;
- numerical backend ownership or precision policy;
- security/privacy/trust boundary;
- model-selection or scientific acceptance rule;
- lifecycle/release governance;
- cross-repository dependency direction.

Method-local implementation details that do not change such a decision belong in method documentation or code, not a new ADR.

## Required ADR sections

Each ADR should include:

1. Status and date.
2. Context/problem.
3. Decision.
4. Invariants/acceptance evidence.
5. Consequences and trade-offs.
6. Alternatives considered.
7. Reversal/supersession conditions.
8. References where research/standards materially govern the decision.
