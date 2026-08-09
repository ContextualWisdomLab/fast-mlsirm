# ADR-0001: Reusable Measurement Core and Hosted Product Boundary

- **Status:** accepted
- **Date:** 2026-08-09
- **Decision owners:** fast-mlsirm maintainers

## Context

The repository has expanded from a narrow MLSIRM toolkit into reusable assessment/rubric/scoring contracts, automated-scoring validation, enterprise-issue measurement contracts, generated-item workflows, and additional psychometric diagnostics. At the same time, `ContextualWisdomLab/psychometrics-commons` exists as the downstream hosted assessment product. Without a hard boundary, HTTP/session/consent/database/UI concerns can leak into the numerical library and create a reverse dependency that prevents independent reuse.

## Decision

`fast-mlsirm` is the **domain-neutral measurement and scientific computation layer**. It owns reusable measurement contracts, observations, calibration/model diagnostics, linking/equating, item/rater evidence, factor/model selection, recovery/simulation, generated-item governance primitives, and Rust numerical kernels.

Psychometrics Commons owns hosted HTTP/admin APIs, participant/session/consent/result lifecycle, product persistence/migrations, tenant/resource authorization, end-user UI, deployment composition, and hosted research-release workflows.

Adjacent repositories such as Keyverse, TEPP, Gyeot, semantic-data-portal, contextual-orchestrator, and EgressWeave are explicit integrations, not hidden dependencies.

## Invariants

- `fast-mlsirm` remains independently installable and useful without any CWL hosted service.
- No ORM/database model, HTTP route, session type, consent object, product UI component, or deployment composition from Psychometrics Commons is imported into this repository.
- Cross-repository integration uses versioned contracts or immutable artifacts.
- Generic identity, egress, temporal-event, and LLM-orchestration concerns remain in their owning bounded contexts.

## Consequences

Positive:

- reusable scientific core can serve RAG, automated essay scoring, enterprise issue measurement, and other assessment domains;
- hosted product can evolve independently;
- scientific validation is not coupled to web-stack churn;
- clearer acquisition/security due-diligence boundary.

Costs:

- some product workflows require explicit adapter contracts;
- downstream products must own persistence and authorization rather than asking the library to do it implicitly.

## Rejected alternatives

1. **Rebuild the hosted runtime inside `fast-mlsirm`.** Rejected because it duplicates Psychometrics Commons and destroys library independence.
2. **Move all reusable scoring contracts downstream.** Rejected because the same contracts are useful across independent assessment domains and are part of measurement reproducibility.
3. **Share one product database across repositories.** Rejected because it creates hidden coupling and breaks bounded-context ownership.

## Security and privacy implications

The library should minimize raw sensitive payload retention and use opaque IDs/digests/provenance when sufficient. Tenant authorization, identity linkage, encryption/retention/data-residency, consent, and privileged-access operations remain downstream responsibilities.

## Compatibility / rollback

A future change may supersede this ADR only if a new architectural boundary preserves independent scientific reuse. Moving product-specific runtime code into this repository is not a compatible incremental change.
