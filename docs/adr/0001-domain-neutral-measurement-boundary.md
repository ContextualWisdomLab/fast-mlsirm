# ADR-0001: Domain-neutral measurement boundary

Status: **Accepted**  
Date: 2026-08-09

## Context

`fast-mlsirm` has grown from an MLS2PLM-focused toolkit into a reusable measurement layer with assessment/rubric/scoring contracts, automated-scoring adapters, psychometric diagnostics, item/rater calibration and release evidence. ContextualWisdomLab also has downstream hosted/application bounded contexts, especially Psychometrics Commons.

Without an explicit boundary, the library can accidentally absorb participant/session state, identity, hosted persistence, UI and deployment logic. That would make the numerical core harder to install independently, create circular repository dependencies, and duplicate ownership already assigned to other CWL services.

## Decision

`fast-mlsirm` owns reusable domain-neutral measurement contracts and scientific/numerical capabilities:

- Assessment/Rubric/Scoring contracts and observations;
- calibration, CTT/IRT/MIRT, MLSIRM/MLS2PLM, facets, testlets and related kernels;
- model diagnostics, linking, DIF/invariance/fairness, CAT/ATA and G-theory;
- factor retention/model comparison, bifactor scoreability, rotation and recovery;
- automated-scoring and LLM-judge measurement primitives;
- governed rubric/item-bank primitives and portable scientific/audit reports.

The hosted product boundary belongs downstream. In particular `ContextualWisdomLab/psychometrics-commons` owns hosted/public/admin APIs, participant/session/response/consent/result lifecycle, product databases/migrations, tenant/resource authorization, UI/reference client behavior, deployment composition and research-release orchestration.

The dependency direction is:

```text
hosted/downstream product -> fast-mlsirm
```

and never:

```text
fast-mlsirm -> hosted product
```

Other CWL services are explicit optional integrations, not hidden implementation dependencies.

## Invariants and evidence

- The package must install and execute without Psychometrics Commons source or runtime.
- No product ORM/database, HTTP route, session/consent or UI type may become a required `fast_mlsirm` dependency.
- Cross-repository composition uses versioned APIs/contracts/immutable artifacts.
- Hosted product state must not be recreated under a library-local assessment runtime service.
- `AGENTS.md` and `CLAUDE.md` carry the same boundary.

## Consequences

Benefits:

- independent adoption and testing;
- clear MSA ownership and security authority;
- lower coupling between scientific evolution and product deployment;
- reusable measurement contracts across essay, RAG, enterprise-issue and other domains.

Costs:

- downstream adapters are required for persistence/transport;
- some end-to-end features require cross-repository integration tests rather than one monolith.

## Alternatives considered

1. **Make fast-mlsirm the hosted platform.** Rejected because it couples scientific kernels to application infrastructure.
2. **Keep only numerical functions and move all contracts downstream.** Rejected because versioned measurement/scoring contracts are reusable scientific primitives and must stay adjacent to the numerical interpretation they govern.

## Reversal conditions

Supersede this ADR only if the organization intentionally redefines repository bounded contexts and provides a migration plan preserving independent scientific/numerical reuse.

## References

International Organization for Standardization, International Electrotechnical Commission, & Institute of Electrical and Electronics Engineers. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.
