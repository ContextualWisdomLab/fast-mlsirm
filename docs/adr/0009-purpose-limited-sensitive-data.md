# ADR-0009: Preserve legitimate sensitive-data utility through purpose limitation

- Status: **Accepted**
- Date: 2026-08-09
- Owner: reusable contract/privacy boundary

## Context

Psychometric, enterprise, longitudinal, and human-rating workflows may require exact person, account, source, evidence, group, or occasion linkage. Blanket irreversible masking at every boundary can destroy the joins and longitudinal/contextual relationships required for valid measurement and operational review. Conversely, copying raw PII into every measurement artifact expands unnecessary privacy, breach, and audit scope.

## Decision

`fast-mlsirm` does not adopt blanket PII masking as its default privacy architecture. It uses **purpose-limited data minimization and separated identity/evidence domains**:

- canonical measurement artifacts contain raw sensitive content only when that exact content is scientifically or contractually necessary;
- durable audit/provenance prefers content digests, opaque identifiers, bounded metadata, and source/evidence references over replicated raw text;
- hosted identity resolution, participant/account linkage, residency, encryption keys, retention/deletion, and data-subject workflows belong to the system that owns those records, normally Psychometrics Commons or another domain service;
- authorization and selective disclosure govern access to re-identifiable evidence rather than pretending that a masked value is still operationally equivalent; and
- downstream reports disclose only the minimum evidence required for their approved purpose while preserving exact scientific values where the report contract requires them.

This decision is a privacy architecture, not a waiver of law, consent, security, or organizational policy.

## Invariants

1. A measurement computation must not require broad PII replication merely for convenience.
2. A feature that genuinely requires exact longitudinal/context linkage documents the purpose and owning system.
3. Durable content digests are not treated as anonymous when they can be linked back to identifiable source records.
4. Provider/model calls receive sensitive content only through an explicit authorized boundary; provider exception text is not copied into durable audit evidence.
5. Authorization, encryption, retention, and deletion remain enforceable by the data-owning service even when measurement artifacts are preserved.
6. Group/fairness/DIF analysis uses the minimum protected attributes required by the approved analysis and preserves governance around their interpretation.
7. Security incidents or legal requirements can revoke access to source content without invalidating content-addressed measurement metadata that no longer contains the source itself, where policy permits retention of that metadata.

## Alternatives considered

### Mask all PII before every computation

Rejected as a universal rule because it can make repeated, hierarchical, multiple-membership, longitudinal, adjudication, or evidence-trace workflows scientifically or operationally unusable.

### Keep raw records in every artifact for perfect reproducibility

Rejected because it multiplies sensitive data and unnecessarily expands access and breach scope.

### Centralize all identity inside fast-mlsirm

Rejected by ADR-0001. The reusable core should consume authorized opaque/linkage contracts, not become the identity database.

## Failure and recovery

If an operation lacks an authorized purpose or cannot satisfy the owning system's access policy, it fails rather than substituting fabricated/masked values that change the statistical design. Sensitive source access can be revoked independently of immutable non-content provenance where the applicable retention policy allows it.

## Compatibility and rollback

Data contracts identify whether a field is content, digest, opaque reference, or governed metadata. A change that introduces new raw sensitive fields requires privacy/security review and a new schema version. Removing raw fields should provide a migration path preserving required provenance.

## Verification

Privacy tests should cover source-free audit outputs, provider-exception redaction, metadata injection, replay/cross-identity failures, bounded payloads, and absence of uncontrolled raw source/PII in durable result artifacts. Hosted end-to-end authorization/retention tests belong to the downstream owner.

## Standards basis

The implementation should be mapped to the applicable Korean privacy/regulatory requirements and customer control environment by the data-owning hosted product. The reusable architecture also supports audit/control evidence expected by frameworks such as SOC 2 and public-sector/cloud security reviews, without claiming certification from this ADR.

## Consequences

This architecture requires stronger identity/reference governance than blanket masking, but preserves legitimate measurement utility while minimizing unnecessary proliferation of sensitive content.
