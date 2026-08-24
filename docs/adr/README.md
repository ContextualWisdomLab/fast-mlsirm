# fast-mlsirm Architecture Decision Records

This directory is the authoritative decision log for architecture, scientific interpretation, trust boundaries, and cross-repository ownership decisions that materially affect `fast-mlsirm`.

Template for new material decisions: `docs/adr/0000-template.md`. The template is guidance, not a live ADR and is therefore intentionally excluded from the decision index below.

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
| [0009](0009-adaptive-rotation-selection.md) | Accepted | Protected main uses a Rust criterion registry, deterministic multi-start and criterion-neutral empirical selection; no universal best criterion or global-optimum claim. GPU/additional-criterion expansion remains separately gated. |
| [0010](0010-llm-orchestration-and-credentials.md) | Accepted | Model-backed automation uses provider-neutral boundaries, NVIDIA NIM credentials where needed, and never uses Copilot credentials for development scheduling. |
| [0011](0011-canonical-pyo3-public-export-registry.md) | Proposed | Future Rust-backed features converge on one reviewed PyO3/public-export registry instead of competing extension initializers/import rewrites. |
| [0012](0012-purpose-limited-sensitive-data.md) | Accepted | Preserve valid measurement linkage through purpose-limited sensitive-data handling rather than blanket masking or raw-data proliferation. |
| [0013](0013-continuous-execution-and-documentation-governance.md) | Proposed | Keep autonomous work work-conserving and enforce one canonical cross-cutting documentation writer with explicit maturity states. |
| [0014](0014-bounded-llm-judge-category-inputs.md) | Proposed | Bound LLM-judge category inputs to exact built-in scalars and keep model/provider security evidence fail-closed and independently verifiable. |
| [0015](0015-multi-item-irt-fit-boundary.md) | Proposed | Enforce the multi-item dichotomous/polytomous contract at public IRT fitters and require explicit readiness evidence before interpreting estimates. |
| [0016](0016-figma-buyer-evidence-design-boundary.md) | Accepted | Bind the buyer-review Figma file ID to repository-local packet validation while keeping Code Connect and hosted UI ownership downstream. |
| [0017](0017-bradley-terry-mm.md) | Accepted | Adopt Bradley–Terry fitted by Hunter MM, plus the implemented additive-ties BRATT variant; do not claim Rao–Kupper/Davidson. |
| [0018](0018-angoff-delta-plot-dif.md) | Accepted | Adopt Angoff delta-plot (Magis & Facon threshold) as the small-sample observed-score DIF screen; distinct from MH/logistic/SIBTEST. |
| [0019](0019-rust-longitudinal-state-engine.md) | Proposed | Rust owns the first respondent-level longitudinal state layer as independent OLS trends and caller-supplied discrete AR; full joint multilevel estimation remains gated. |
| [0020](0020-joint-hierarchical-ctar-rasch.md) | Proposed | Joint MAP hierarchical continuous-time AR(1) Rasch estimates shared `(mu, tau, lambda)` and person-occasion states; MMMC and GPU parity remain excluded. |

## ADR completeness rule

A material decision should have an ADR when it changes one or more of:

- repository/bounded-context ownership;
- public serialized contract or versioning rule;
- psychometric model parameterization/identification/interpretation;
- numerical backend ownership or precision policy;
- PyO3/native binding/public-export authority;
- security/privacy/trust/credential boundary;
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
7. Failure/degraded/recovery behavior where applicable.
8. Security/privacy implications where applicable.
9. Compatibility/migration/rollback and reversal/supersession conditions.
10. Verification/release evidence and references where research/standards materially govern the decision.

## Consistency rule

Accepted ADRs, protected-main code/tests, `docs/PRD.md`, `docs/TRD.md`, root architecture, UML/ERD, the reusable-core threat model, requirements traceability and release evidence must not contradict one another. A changed accepted decision is superseded through a new ADR rather than silently rewriting history.
