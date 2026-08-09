# ADR-001: Keep fast-mlsirm as the domain-neutral measurement core

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

The repository increasingly serves automated scoring, RAG evaluation, rubric/item generation, model comparison, enterprise issue measurement, and downstream Psychometrics Commons use. Recreating hosted product concerns here would couple numerical/scientific release cadence to identity, session, consent, tenant, database, UI, and deployment concerns.

## Decision

`fast-mlsirm` owns reusable measurement contracts and psychometric computation. Hosted lifecycle and product persistence are downstream concerns.

It owns:

- Assessment/Rubric/Scoring contracts;
- item/rater observations and provenance;
- calibration and psychometric numerical kernels;
- fit, DIF/invariance, reliability/scoreability, linking, recovery and model-selection evidence;
- provider-neutral authoring/scoring interfaces;
- deterministic reports and release evidence.

It does not own:

- identity/RBAC/tenant administration;
- assessment session/consent/result-access lifecycle;
- hosted persistence/migrations;
- billing/support CRM;
- general LLM routing credentials;
- hosted UI/deployment control plane.

`ContextualWisdomLab/psychometrics-commons` is the canonical hosted psychometrics product and consumes this package downstream. Other CWL repositories remain optional bounded integrations.

## Consequences

- The package stays independently installable and testable.
- Downstream systems must integrate by versioned contracts/artifacts rather than importing internals.
- Product-specific fields cannot be added merely because one consumer needs them.
- A hosted runtime must not be rebuilt inside this repository.

## Alternatives considered

1. **Monolith including hosted runtime** — rejected because it destroys modularity, ownership clarity, and standalone reuse.
2. **Thin numerical crate only** — rejected because versioned Assessment/Rubric/Scoring contracts are reusable domain assets and belong with the measurement semantics they govern.

## References

ISO/IEC/IEEE. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.
