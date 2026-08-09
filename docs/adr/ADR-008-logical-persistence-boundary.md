# ADR-008: The core owns logical information contracts, not hosted persistence

- Status: Accepted
- Date: 2026-08-09
- Deciders: ContextualWisdomLab maintainers

## Context

Assessment, rubric, item-bank, scoring, calibration, context-membership, and temporal objects have relationships that are useful to document as an ERD. Adding an ORM or physical schema to `fast-mlsirm`, however, would conflict with its standalone-library role and duplicate persistence already owned by downstream products.

## Decision

`fast-mlsirm` defines logical entities, identifiers, relationships, versions, and provenance semantics. It does not require a database or own hosted persistence/migrations.

`docs/architecture/ERD.md` is therefore a **logical contract ERD**. A downstream service may map those entities into relational/document/event storage provided it preserves the relevant semantics:

- logical ID distinct from content fingerprint;
- immutable/versioned historical interpretation;
- exact rubric/task/item/engine/model revision links;
- descriptive two-or-more-word `snake_case` database object names by default;
- opaque durable public identifiers rather than sequential numeric public IDs where externally visible identity is needed;
- raw sensitive content only under the owning service's purpose-bound access, encryption, retention, deletion/export, and audit controls.

## Consequences

- The package has no required database driver or migration framework.
- Integration tests can validate serialized logical contracts without provisioning a database.
- Psychometrics Commons or another product can choose Postgres/event/object storage independently.
- Data residency, tenant isolation, PII retention, consent, and data-rights workflows remain product responsibilities.

## Alternatives considered

1. **Canonical Postgres schema in fast-mlsirm** — rejected because it couples the library to one hosted persistence architecture.
2. **No ERD at all** — rejected because logical relationships and provenance cardinalities are important for downstream correctness.
3. **Duplicate product persistence models** — rejected because it creates schema divergence and reverse coupling.

## References

ISO/IEC/IEEE. (2022). *ISO/IEC/IEEE 42010:2022 Software, systems and enterprise—Architecture description*.
