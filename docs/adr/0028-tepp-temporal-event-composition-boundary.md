# ADR-0028: TEPP temporal/event composition boundary

Status: Accepted
Date: 2026-09-01

## Context

`fast-mlsirm` already contains time-indexed psychometric numerical kernels, including the Rust joint-MAP hierarchical continuous-time AR(1)/OU Rasch implementation in `crates/mlsirm-core/src/longitudinal_irt.rs`. Earlier proposed ADRs described those numerical models in language broad enough to imply ownership of temporal-event semantics.

That wording conflicts with the ContextualWisdomLab bounded-context map. TEPP is the canonical owner of temporal/event analysis. `fast-mlsirm` is the canonical reusable psychometric numerical engine. A numerical likelihood may consume elapsed time or occasion indices without becoming the owner of the event ontology that gives those carriers meaning.

## Decision

**TEPP owns temporal/event composition and semantics.** This includes event ontology and graph construction, valid/event-time meaning, changing-membership history, temporal split and leakage policy, and domain interpretation of change or drift.

**fast-mlsirm owns reusable time-indexed psychometric numerical kernels.** It may implement and validate likelihoods, state-transition arithmetic, gradients, optimization, uncertainty, recovery, and parity for an explicit psychometric model when time/occasion carriers are already supplied under a typed contract. It may validate estimator-local numerical invariants such as finite ranges, monotone offsets required by the declared likelihood, positive elapsed intervals, and bounded shapes.

Those checks do not authorize `fast-mlsirm` to infer event meaning, reconstruct a TEPP event graph, decide temporal validity, derive changing-membership semantics, or define leakage-safe temporal partitions.

Cross-context integration uses an Anti-Corruption Layer. A TEPP-originated occasion/temporal design is admitted only through an explicit versioned contract or adapter. No cross-service SQL, TEPP database access, or hidden runtime dependency is permitted.

The existing protected-main CT-AR Rasch estimator remains a `fast-mlsirm` numerical psychometric kernel. Its equations and recovery evidence are not deprecated by this decision. Proposed ADR-0007, ADR-0019, and ADR-0020 remain useful numerical/model-design records but are interpreted through this ownership boundary wherever their wording is broader.

## Invariants and acceptance evidence

- `crates/mlsirm-core` remains the production owner of psychometric arithmetic; Python remains validation, marshalling, orchestration, reporting, and explicit reference testing.
- Time-indexed numerical kernels must publish exact equations, identification conditions, estimator identity, numerical bounds, and true-parameter recovery evidence appropriate to the claim.
- Temporal/event semantics must enter through a typed foreign-context contract; a raw timestamp or display label is not sufficient authority to reconstruct TEPP semantics.
- Measurement occasion is allowed as an explicit facet/value without transferring temporal-event ownership.
- Dynamic or longitudinal validation must distinguish numerical model recovery from TEPP-owned temporal split/leakage and event-validity evidence.
- Existing CT-AR identities such as `joint_map_hierarchical_ctar_rasch` and `continuous_time_ar1_ou` remain formulation identifiers, not event-ontology identifiers.
- No provider SDK, credential, or LLM orchestration is introduced by this boundary. Contextual-orchestrator remains the sole LLM orchestration layer.

## Context Graph and EA projection

Architecture/package/backend/toolchain/consumer-lifecycle facts may be projected to `ContextualWisdomLab/enterprise-architecture-core` only through an immutable released `ContextualWisdomLab/context-graph-contracts` versioned Context Assertion / CloudEvent / conformance contract with provenance.

Estimator values, latent scores, DIF/fit diagnostics, and scientific-validity evidence are not authoritative EA facts and must not be duplicated into the architecture decision plane.

At adoption time, neither foreign repository exposes an immutable GitHub release. Therefore production EA projection fails closed rather than pinning an unreleased sibling PR head. Their source, docs, refs, and PR lifecycle remain owned by the Context Fabric writer.

## Consequences

The repository can retain scientifically justified time-indexed numerical psychometric methods without becoming a competing temporal platform. New longitudinal methods need both psychometric evidence and a clean TEPP integration boundary when their inputs depend on temporal-event semantics.

Canonical architecture/PRD/TRD/traceability wording that still conflates numerical temporal kernels with temporal-event ownership is tracked as documentation drift and must be aligned in reviewed follow-up work. This ADR is the governing ownership rule during that migration.

## Alternatives considered

1. **Move every time-indexed numerical kernel to TEPP.** Rejected because psychometric likelihood, parameter recovery, information, and estimator arithmetic belong to the canonical psychometric numerical engine.
2. **Let fast-mlsirm own both numerical models and temporal-event semantics.** Rejected because it duplicates TEPP's bounded context and creates incompatible event truth.
3. **Treat timestamps as generic scalars with no boundary contract.** Rejected because numerical admissibility does not establish event meaning, validity, or leakage-safe longitudinal design.

## Failure, migration, and reversal

If a requested model requires temporal semantics that cannot be represented by the released integration contract, fail closed with an unsupported-contract result rather than inferring or coercing the missing semantics.

No existing numerical public API is removed by this ADR. Future API revisions should make foreign temporal provenance and formulation identity explicit without changing parameter meaning silently. Reversal requires a later Accepted ADR and coordinated TEPP migration evidence.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Jeon, M., & Rabe-Hesketh, S. (2016). An autoregressive growth model for longitudinal item analysis. *Psychometrika, 81*(3), 830–850. https://doi.org/10.1007/s11336-015-9489-2

Laird, N. M., & Ware, J. H. (1982). Random-effects models for longitudinal data. *Biometrics, 38*(4), 963–974. https://doi.org/10.2307/2529876
