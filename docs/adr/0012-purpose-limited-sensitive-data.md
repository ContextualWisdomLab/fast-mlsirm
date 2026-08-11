# ADR-0012: Preserve measurement utility through purpose-limited sensitive-data handling

Status: **Accepted**  
Date: 2026-08-09

## Context

Psychometric, enterprise, longitudinal and human-rating workflows can require exact participant/source/group/context/occasion linkage. Blanket masking at every boundary can destroy repeated-measure, multilevel, multiple-membership, adjudication, DIF/fairness and evidence-trace relationships. At the same time, duplicating raw PII or sensitive source text into every calibration/report/provenance artifact unnecessarily increases privacy, breach and audit scope.

## Decision

`fast-mlsirm` does **not** use blanket PII masking as its default reusable-core privacy architecture. It uses purpose limitation, data minimization, separated identity/evidence domains, authorization, bounded retention interfaces, selective disclosure and exact provenance.

- Raw sensitive content enters a computation only when the caller's authorized scientific/business purpose requires it.
- Durable reusable artifacts prefer opaque ids, content digests, bounded metadata and governed source references over copied raw text.
- Hosted participant/account identity resolution, consent, tenant authorization, encryption keys, residency, retention/deletion and data-subject workflows remain owned by Psychometrics Commons or the appropriate downstream data-owning service.
- Protected attributes used for DIF/fairness remain governed inputs; they are not generalized into unrestricted report metadata.
- Provider/model calls receive sensitive data only across an explicit authorized provider boundary, and provider exception text is not copied into durable audit evidence.
- If the measurement design requires exact linkage and that linkage is not authorized/available, the operation fails rather than silently flattening or substituting masked pseudo-values that change the estimand.

A digest or pseudonymous identifier is not automatically treated as anonymous merely because plaintext PII is absent.

## Invariants / evidence

1. Canonical measurement/result artifacts omit raw sensitive source text unless their public contract explicitly requires it.
2. Source-free audit/provenance outputs retain enough immutable identity to reconstruct authorized evidence without copying it into every artifact.
3. Cross-object identity/replay checks prevent one person's/source's evidence from being rebound to another artifact.
4. Logging and error paths do not echo provider credentials, arbitrary source content or uncontrolled PII.
5. Downstream hosts can revoke/expire source access according to policy without requiring historical non-content scientific fingerprints to be rewritten, where applicable law/policy allows those fingerprints to remain.
6. Any new raw sensitive field in a canonical contract requires versioned schema/privacy review.
7. No documentation may represent masking removal as exemption from legal, contractual, consent or security obligations.

## Consequences and trade-offs

This architecture is more demanding than blanket redaction: data ownership, access and purpose must be explicit. It preserves legitimate scientific/operational utility while minimizing unnecessary proliferation of sensitive content.

## Alternatives considered

### Mask every identifier/value before measurement

Rejected as a universal design because it can invalidate longitudinal, hierarchical, multiple-membership, rater, adjudication and evidence-trace workflows.

### Persist all raw evidence for maximum reproducibility

Rejected because it expands sensitive-data scope far beyond the minimum needed for reproducible measurement.

### Move identity into fast-mlsirm

Rejected by ADR-0001. The reusable core consumes governed identifiers/references; it does not become the hosted identity database.

## Reversal / supersession conditions

A superseding decision is required if the reusable package itself begins owning durable participant/customer identity or hosted retention/deletion lifecycle. That would also require reconsidering ADR-0001 and the logical persistence boundary.

## Standards/control basis

This ADR is designed to support evidence for privacy/security management and SOC 2/CSAP-oriented controls without claiming certification. The exact legal/privacy requirements and control mapping are owned by the data controller/host and must be kept current for its jurisdiction and customer obligations.
