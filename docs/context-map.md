# fast-mlsirm Context Map

Status: Active architecture contract
Owner: `ContextualWisdomLab/fast-mlsirm`
Governing temporal boundary: ADR-0028

## Purpose

`fast-mlsirm` is the reusable psychometric numerical engine. This map separates model meaning, numerical execution, evidence, transport, hosted-product concerns, temporal/event semantics, and enterprise-architecture authority so a new family or integration does not become a cross-context shortcut.

Production psychometric arithmetic is Rust-owned. Python is limited to validation, immutable marshalling, reporting, and binding ergonomics; it must not become a second likelihood, scoring, matrix/vector, recovery, or optimization implementation.

## Internal bounded contexts

| Bounded context | Owns | Does not own |
| --- | --- | --- |
| `Model Specification` | Typed model/formulation identity, parameter blocks, supported/research-candidate/unsupported status, identification and recovery requirements. | Likelihood arithmetic, provider routing, event ontology. |
| `Estimation` | Rust likelihood/gradient/integration/optimization kernels and estimator-local convergence contracts for admitted specifications. | Hosted workflow/session state, TEPP temporal semantics. |
| `Scoring` | Rust-owned score/information/EAP or other explicitly supported scoring kernels over released model contracts. | Participant lifecycle, UI interpretation, causal/business outcome claims. |
| `Diagnostics` | Fit, dependence, DIF/invariance/fairness and other measurement diagnostics whose estimands are explicitly defined. | EA facts, temporal event meaning, generic product analytics. |
| `Simulation-Recovery` | Known-truth simulation, deterministic seed manifests, bias/MAE/RMSE/coverage, Monte Carlo uncertainty, identifiability/recovery evidence and reproducibility gates. | Production respondent/session records or architecture inventory. |
| `Compute Backend` | Rust CPU parallelism, GPU kernels where promoted, CPU/GPU parity, deterministic execution/resource contracts. | Model semantics or provider/model routing. |
| `Public Binding` | Stable Rust/PyO3/Python API contracts, validation, immutable marshalling, installed-package behavior, reporting and compatibility/version surfaces. | Independent production statistical arithmetic or hidden fallback estimators; governed explicit reference/parity calculations remain permitted. |

### Internal dependency direction

```text
Model Specification
      │
      ├──────────────┬───────────────┐
      ▼              ▼               ▼
 Estimation       Scoring        Diagnostics
      │              │               │
      └───────┬──────┴───────┬───────┘
              ▼              ▼
      Simulation-Recovery  Compute Backend
              └──────┬───────┘
                     ▼
                Public Binding
```

The diagram is an ownership dependency map, not a runtime call graph. A public binding may expose contracts from the other contexts, but it may not reimplement their numerical rules. Explicit NumPy/reference surfaces that exist solely for governed validation or CPU/GPU/parity comparison remain reference evidence rather than production numerical ownership.

## Foreign bounded contexts and relationship contracts

### TEPP — temporal/event authority

TEPP owns temporal/event composition and semantics: event ontology and graph construction, valid/system/event-time meaning, event ordering, changing-membership history, temporal split/leakage policy, and domain interpretation of change or drift.

`fast-mlsirm` may own reusable time-indexed psychometric numerical kernels when the occasion/time carriers and exact model equations are already supplied. A TEPP-originated temporal design crosses an explicit versioned immutable Anti-Corruption Layer with compatibility identity and provenance. Direct TEPP database access, hidden TEPP runtime dependencies, and cross-service SQL is prohibited.

The research interpretation of this split is maintained in [`docs/traceability/temporal-research-ownership.md`](traceability/temporal-research-ownership.md). The cited longitudinal and multilevel papers ground numerical/model questions; they do not transfer TEPP's temporal/event semantic authority.

Relationship: **TEPP upstream semantic owner → Anti-Corruption Layer → fast-mlsirm numerical consumer**.

### psychometrics-commons — hosted product

psychometrics-commons is a downstream hosted-product consumer. It owns participant/session/consent/result lifecycle, product HTTP/admin APIs, product persistence and migrations, resource authorization, reference clients, research-release orchestration, deployment composition, and product UI.

`fast-mlsirm` remains independently installable and exposes reusable versioned measurement/numerical contracts. Product ORM/database models, HTTP types, UI state, deployment configuration, and hosted runtime behavior do not move into this repository.

Relationship: **fast-mlsirm upstream reusable engine → released public contract → psychometrics-commons downstream product**.

### context-graph-contracts — Context Fabric Shared Kernel

context-graph-contracts is the contract-only Shared Kernel for canonical object/authority references, truth status/origin, valid/system time, provenance, Context Assertions, CloudEvents, schema/conformance, and admission.

This repository consumes that owner surface read-only. Architecture/package/backend/toolchain/consumer-lifecycle facts may cross the boundary only through an immutable released context-graph-contracts contract with provenance. An unreleased sibling branch or PR head is not production authority.

Relationship: **released Shared Kernel contract → fast-mlsirm integration adapter**.

### enterprise-architecture-core — EA Decision Plane

enterprise-architecture-core is the authoritative EA Decision Plane. `fast-mlsirm` may project eligible architecture facts only after the Context Graph contract gate above is satisfied. Estimator values, latent scores, DIF/fit diagnostics, and scientific-validity evidence are not EA-authoritative facts and must not be duplicated into that decision plane.

Relationship: **fast-mlsirm eligible lifecycle/architecture facts → released Context Assertion/CloudEvent contract → enterprise-architecture-core**.

## Integration invariants

- No cross-service SQL or direct foreign-context database reads.
- No source, documentation, branch, PR-state, or release mutation in Context Fabric repositories from this repository's writer lane.
- No unreleased sibling SHA becomes a stable public dependency or EA authority.
- No estimator result, latent score, DIF/fit diagnostic, recovery metric, bias/RMSE/coverage result, or scientific-validity evidence is promoted as an EA architecture fact.
- No TEPP event ontology or event-time policy is reconstructed inside `fast-mlsirm` merely because a numerical model consumes time or occasion carriers.
- New model/dependence families enter through `Model Specification`; support is promoted only with explicit equations/formulation identity, identification, Rust implementation, primary research grounding, and required recovery evidence.
- Public bindings fail closed when a requested model/estimator/backend/integration contract is unsupported; they do not silently substitute a different scientific estimand.

## Change rule

A change that moves responsibility across one of these boundaries requires an explicit architecture decision and matching fitness tests before implementation. Historical research/model ADRs remain evidence of their numerical design; they are interpreted through the current owner boundary rather than rewritten to erase valid scientific history.
