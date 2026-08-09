# ADR-NNNN — <Decision title>

- **Status:** Proposed | Accepted | Deprecated | Superseded | Rejected
- **Date:** YYYY-MM-DD
- **Decision owner:** <repository/service/bounded context>
- **Implementation status:** <not started | partial | active | complete | retired>
- **Supersedes:** <ADR or none>
- **Superseded by:** <ADR or none>

## Context and decision drivers

Describe the concrete problem, current evidence, constraints, assumptions, affected users/systems and why a durable architectural decision is required. Distinguish a current protected-main fact from a proposed future capability.

## Decision

State exactly what is chosen. Name the owning repository/service for every responsibility and the allowed dependency direction. Name forbidden coupling explicitly.

## Contract and interface consequences

When affected, specify:

- public API and schema/version behavior;
- identifiers and content/provenance identity;
- idempotency and replay semantics;
- ordering/time semantics;
- timeout/retry/resource limits;
- error/fail-closed behavior;
- compatibility/deprecation window.

If not applicable, say why.

## Numerical and scientific consequences

When mathematical/psychometric behavior is affected, specify:

- model/parameterization and identification assumptions;
- numerical source of truth;
- CPU/GPU ownership and parity requirements;
- multilevel/cross-classified/multiple-membership/testlet/temporal implications;
- interpretation and validity limits;
- recovery, bias/RMSE/coverage and model-selection evidence required.

If not applicable, say why.

## Invariants and controls

List properties that must remain true and the test, check, review or operational control that proves each one.

## Failure, degraded and recovery behavior

Define fail-closed behavior, unsupported/indeterminate states, bounded retries, poison/untrusted input handling, rollback triggers and recovery semantics. Do not convert absent evidence into success.

## Security, privacy and compliance consequences

Address, as applicable:

- authentication/authorization and separation of duties;
- tenancy/resource authority;
- sensitive/PII data purpose and minimum necessary disclosure;
- encryption/key boundaries;
- retention/erasure/export/residency;
- audit/provenance/tamper evidence;
- software supply chain and credential scope;
- CSAP/SOC 2/AI governance control ownership without claiming certification.

## Migration, compatibility and rollback

Explain how existing users/artifacts migrate, what remains backward compatible, how historical evidence is preserved, and how to reverse the change safely. Include objective rollback triggers.

## Verification and acceptance evidence

List measurable evidence required before the decision is considered implemented. Include exact-head/release-artifact binding where relevant and realistic domain tests rather than nominal smoke only.

## Alternatives considered

For each materially distinct alternative, state why it was accepted/rejected and what evidence would change that judgment.

## Accepted risks and follow-up work

List residual risks, deferred implementation, external dependencies and issue/PR triggers. An accepted ADR must not hide unfinished work by describing it as complete.

## Sources

Use current authoritative standards, official primary technical documentation and primary peer-reviewed papers where material. Format bibliographic entries in APA 7 style. A citation does not replace implementation or empirical validation evidence.

## Supersession / reversal criteria

State objective conditions under which this ADR should be superseded or reversed.
