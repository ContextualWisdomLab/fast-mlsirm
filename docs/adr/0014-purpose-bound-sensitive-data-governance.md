# ADR 0014: Purpose-bound sensitive-data governance without blanket masking

- **Status:** Proposed
- **Date:** 2026-08-09
- **Decision owners:** fast-mlsirm maintainers
- **Scope:** Reusable psychometric contracts, audit provenance, model execution, and downstream integration

## Context

Psychometric, longitudinal, multiple-membership, rater, adjudication, and fairness analyses often require stable linkage across respondents, tasks, raters, contexts, groups, and occasions. Blanket removal or irreversible masking of every potentially identifying field can destroy those relationships, prevent correction of duplicate or erroneous records, block lawful audit and erasure workflows, and make consequential validation impossible.

At the same time, ambient propagation of raw names, response text, prompts, evidence, external provider output, or identity mappings through logs and model requests creates privacy and security risk. `fast-mlsirm` is a reusable computation and contract library, not the owner of tenant identity, consent, legal basis, customer retention policy, or hosted authorization.

## Decision

### Core-library data minimization

The core library stores and transmits descriptive opaque identifiers, content fingerprints, bounded metadata, and explicit references rather than raw source text wherever numerical computation does not require the raw value. Raw responses, prompts, evidence, and identity mappings remain outside governed scoring, calibration, and audit artifacts unless a specific public API explicitly accepts them for immediate in-process computation.

SHA-256 fingerprints and public handles provide content identity and replay detection. They are not authentication credentials, authorization decisions, signatures, consent records, or proof of lawful processing.

### Downstream ownership

Hosted bounded contexts, including Psychometrics Commons and domain applications, own:

- tenant isolation;
- authentication and authorization;
- participant and identity lifecycle;
- consent, legal basis, contractual purpose, and policy metadata;
- durable persistence and migrations;
- data-residency, retention, export, correction, and erasure workflows;
- encryption keys and identity-map custody;
- incident response and regulatory evidence.

`fast-mlsirm` must not acquire a reverse dependency on those product services or their ORM, HTTP, database, deployment, or UI types.

### Preferred controls

Where identity linkage is necessary, systems should prefer combinations of:

- purpose-bound RBAC or ABAC decisions;
- opaque, nonnumeric subject and artifact identifiers;
- pseudonymous analytical identifiers separated from encrypted identity mappings;
- field- or envelope-level encryption with KMS-backed key ownership;
- selective disclosure and scoped decryption rather than ambient plaintext;
- tenant and residency partitioning in the downstream host;
- bounded retention, export, and deletion policies;
- access-purpose and data-lineage audit events;
- tamper-evident provenance over immutable contract and result fingerprints;
- explicit consent, legal-basis, and policy references where the host owns them;
- provider allowlists and prompt/source minimization for model-backed execution.

Blanket masking is not the default control when it invalidates the measurement or required operational workflow. Any unmasked use must remain purpose-limited, authorized, encrypted where appropriate, auditable, and bounded in retention and disclosure.

### Logging and reporting

Public failures expose stable error codes and caller-independent paths without rejected values, raw source content, credentials, or provider payloads. Reports should expose exact numerical evidence and contract provenance while avoiding raw identity and source content unless the authorized downstream host deliberately adds it.

### Test obligations

Applicable changes require realistic tests for:

- raw-content absence from governed request, observation, calibration, report, and failure artifacts;
- stable linkage through opaque identifiers and fingerprints;
- rejection of duplicate, malformed, cross-revision, or replayed provenance;
- tenant-bound authorization and cross-tenant denial in the downstream host;
- key rotation, retention, export, erasure, and audit replay where implemented;
- prevention of credentials and restricted fields entering model prompts or logs;
- migration and rollback without losing lawful lineage or measurement linkage.

## Consequences

### Positive

- The library preserves the linkage required for multilevel, longitudinal, rater, fairness, audit, and adjudication work.
- Raw identity and source content do not become ambient core-library state.
- Privacy and security responsibility remains with the bounded context that possesses the necessary identity, tenant, legal, and operational knowledge.
- The same contracts can operate standalone or across an MSA boundary.

### Costs and limitations

- Downstream products must implement real authorization, encryption, retention, residency, and identity-map controls.
- Pseudonymization does not make data anonymous and does not eliminate re-identification risk.
- Hashes of low-entropy values can be guessable; callers must not hash raw identifiers as a substitute for pseudonymization.
- This ADR supports CSAP and SOC 2 readiness considerations but does not claim certification or legal compliance.

## Rejected alternatives

1. **Mask or delete every identifier before analysis.** Rejected because it can make multiple-membership, longitudinal linkage, correction, erasure, and audit impossible.
2. **Store raw PII in core scoring and calibration artifacts for convenience.** Rejected because it creates hidden coupling and unnecessary breach surface.
3. **Treat content hashes as authorization or anonymity.** Rejected because hashes provide neither property.
4. **Centralize all identity and consent logic in fast-mlsirm.** Rejected because the reusable numerical library lacks tenant and legal context and would become a hosted product.

## References

National Institute of Standards and Technology. (2020). *NIST Privacy Framework: A tool for improving privacy through enterprise risk management, version 1.0*.

National Institute of Standards and Technology. (2024). *Cybersecurity Framework (CSF) 2.0*.

International Organization for Standardization & International Electrotechnical Commission. (2023). *ISO/IEC 42001:2023 Information technology—Artificial intelligence—Management system*.
