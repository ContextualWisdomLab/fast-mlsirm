# ADR 0013: Continuous execution and canonical documentation governance

Status: **Proposed**  
Date: 2026-08-09  
Decision owners: fast-mlsirm maintainers  
Scope: Repository development loop, architecture documentation, and release evidence

## Context

`fast-mlsirm` combines Rust numerical kernels, Python contracts and orchestration, psychometric diagnostics, automated-scoring interfaces, Rubric-to-item authoring, and enterprise evidence generation. Durable decisions have historically appeared across source code, feature documents, pull-request bodies, issues, agent instructions, and research notes. That distribution creates two risks:

1. an execution loop can stop after describing a blocker or completing one small action even though other safe work exists; and
2. two documentation branches can independently claim authority over PRD, TRD, architecture, ADR, UML, ERD, threat-model, or traceability state.

Both failures increase integration latency and allow shipped behavior, active-PR behavior, and roadmap intent to be confused.

## Decision

### Work-conserving execution

Every autonomous invocation maintains a fresh executable queue and treats commits, merges, review requests, CI reruns, RCA conclusions, documentation updates, and issue closures as intermediate events. After each event the loop selects the next safe action. A blocked merge or external review blocks only that action. The loop ends only when the finite invocation budget is exhausted or all remaining work is non-actionable under current authority.

The loop writes only `ContextualWisdomLab/fast-mlsirm`. Repositories with dedicated writer loops are read-only dependencies. Before every write, the exact branch head, live base tip, target blob or ref, relevant review state, and active-writer evidence are refreshed. Source movement or an active writer makes only the affected branch read-only for the remainder of the invocation.

### One canonical documentation writer

At most one active branch may serve as the canonical cross-cutting documentation authority. It owns changes to:

- `docs/PRD.md`;
- `docs/TRD.md`;
- root `ARCHITECTURE.md`;
- `docs/adr/` and its index;
- UML, logical ERD, threat model, and verification/validation views;
- requirements and research traceability;
- documentation coverage and maturity state.

A competing documentation branch must either contribute unique content to the canonical branch and close, or explicitly supersede the canonical branch with recorded lineage. Parallel authority is prohibited.

### Documentation maturity vocabulary

Every capability described by canonical documents is labelled or otherwise unambiguously classified as one of:

- **IMPLEMENTED / ACCEPTED:** present on protected main and supported by current evidence;
- **ACTIVE PR:** implemented on an unmerged branch and not shipped;
- **PROPOSED:** accepted design direction without completed implementation;
- **PLANNED:** roadmap work without an accepted implementation contract;
- **DOWNSTREAM:** owned by another bounded context such as Psychometrics Commons;
- **REJECTED / SUPERSEDED:** intentionally not part of the governing design.

Unmerged work must never be promoted to protected-main capability in PRD, TRD, Architecture, README, commercial-readiness, or buyer evidence.

### Documentation completeness

A substantive contract change is documentation-complete only when every applicable artifact is updated:

- public behavior and API documentation;
- PRD and TRD requirements;
- an ADR for durable architecture choices;
- UML or ERD when components, data ownership, cardinality, or lifecycle changes;
- requirements-to-ADR-to-implementation/evidence traceability;
- APA 7 primary-source doctoring and equation-to-source traceability for scientific claims;
- identification, interpretation, migration, and rollback boundaries;
- realistic recovery, benchmark, or operational evidence;
- changelog and version when release semantics change.

Documentation does not replace implementation. Conversely, unresolved architecture ambiguity is a product defect and may be selected as the next executable work item when product branches are blocked.

## Current review finding and remediation

An active review found that the documentation matrix called the Proposed
canonical PyO3/public-export registry the native-entrypoint source of truth
while protected main still used separate initializers and package export
paths. The matrix must state target architecture and protected-main behavior
separately until ADR-0011 is implemented; the active PR applies that wording
correction.

## Consequences

### Positive

- Development invocations continue useful work instead of terminating on one blocker.
- PRD, TRD, Architecture, ADR, UML, ERD, threat-model, and traceability views converge on one authority.
- Buyer and reviewer documents cannot silently promote roadmap work to shipped capability.
- Repository-writer conflicts are scoped rather than freezing the entire run.
- Documentation completeness becomes auditable and testable.

### Costs and limitations

- Canonical documentation changes may need rebasing after accepted product PRs merge.
- Mature traceability requires ongoing maintenance rather than a one-time documentation sprint.
- The execution loop cannot manufacture external approval, credentials, permissions, or evidence.
- This ADR governs repository process; it does not authorize automatic merge or release outside normal protection.

## Compliance and verification

Repository tests should fail when required canonical files, maturity vocabulary, ADR status fields, traceability links, or machine-renderable diagram sources disappear. Pull-request review must compare documentation claims with protected-main public exports and exact-head implementation evidence.

## Alternatives considered

1. **Status-report-first automation.** Rejected because it consumes run budget without changing repository state.
2. **One documentation PR per feature with no canonical spine.** Rejected because cross-cutting contracts diverge and supersession becomes circular.
3. **Treat README and PR descriptions as sufficient architecture documentation.** Rejected because neither provides stable decision status, model boundaries, data ownership, or requirements traceability.
4. **Freeze all work while one branch waits for CI or review.** Rejected because unrelated safe actions remain executable.

## References

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2018). *ISO/IEC/IEEE 29148:2018 Systems and software engineering—Life cycle processes—Requirements engineering*.
