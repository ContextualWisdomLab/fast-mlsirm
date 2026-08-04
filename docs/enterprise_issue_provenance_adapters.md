# Enterprise issue provenance adapters

`fast_mlsirm.scoring.enterprise_issue` provides the first provider-neutral boundary for issue #404. This slice deliberately stops at source, evidence-span, and counterevidence provenance. Atomic issue assembly, criterion-observation compilation, semantic extraction, calibration, intervention utility, value of information, queue routing, and reports remain separate reviewable slices.

## Contracts

- `EnterpriseSourceRecord` identifies one exact report, sales-lead record, customer comment, customer complaint, or other enterprise source revision by descriptive identifiers, SHA-256 content/revision fingerprints, bounded character count, and immutable metadata. Source text is never retained.
- `EvidenceSpanRecord` binds an exact half-open source span to the verified source record and requires one explicit epistemic role: directly stated fact, supported inference, counterevidence, unresolved ambiguity, or stakeholder value judgment.
- `CounterevidenceRecord` binds an explicit counterevidence span to the exact fingerprint of the issue statement it challenges.
- `EvidenceSpanRecord.shared_evidence_reference` projects the span into the existing `fast_mlsirm.scoring.EvidenceReference` contract. The adapter does not create a competing scoring-evidence schema.

The five epistemic roles are not sentiment classes. Positive or negative wording is neither stored nor interpreted as consequence, likelihood, urgency, strategic relevance, actionability, priority, or utility.

## Provenance and replay boundary

The contracts are content-addressed and factory sealed. Public handles use a descriptive prefix plus the first 128 bits of the complete SHA-256 contract fingerprint; the complete fingerprint remains available for replay and collision checking. Equivalent metadata key order produces the same identity, while source revision, assertion role, span offsets, and content digests participate in identity.

The boundary is compatible with the entity/derivation/agent separation in W3C PROV-DM, but it is not a complete PROV serialization. A later interoperability slice may add an explicit PROV export without changing these package contracts. W3C PROV treats provenance as information about entities, activities, and agents that can support quality, reliability, and trust assessments; it does not make the underlying claim true by itself (Moreau & Missier, 2013).

## Security and privacy

- Raw source, response, prompt, provider output, and essay fields are rejected from metadata by the shared bounded metadata validator.
- Identifiers must use descriptive two-or-more-token lower `snake_case`.
- Character counts and offsets are bounded; offsets must define a non-empty forward span.
- Direct dataclass construction is rejected so callers cannot bypass source binding or assertion-role validation.
- Validation errors use stable codes and do not reflect rejected caller content.
- The contracts make no claim that a source is truthful, that an inference is valid, that counterevidence is decisive, or that an issue is important.

These controls support NIST AI RMF documentation and traceability outcomes, including documenting system limitations, human oversight, scientific integrity, testing, uncertainty, and risk-management decisions. They do not establish AI RMF conformity or authorize consequential automation (Tabassi, 2023).

## Scientific and product boundaries

This slice performs no psychometric or utility arithmetic and therefore adds no Python duplicate of Rust numerical kernels. In particular, it does not:

- collapse evidence strength into material consequence;
- infer sentiment, likelihood, causality, urgency, or intervention value;
- rank issues or stakeholders;
- treat a content fingerprint as validity evidence;
- learn causal effects from observational text;
- authorize automatic action or replace human governance.

The next compatible slice should assemble immutable `AtomicIssueRecord`, `StakeholderPerspective`, and `CandidateIntervention` adapters from these primitives, then compile criterion-level observations into the existing shared scoring contracts.

## References

Moreau, L., & Missier, P. (Eds.). (2013, April 30). *PROV-DM: The PROV data model* (W3C Recommendation). World Wide Web Consortium. https://www.w3.org/TR/prov-dm/

Python Software Foundation. (n.d.). *dataclasses—Data classes*. Python 3.13.9 documentation. Retrieved August 5, 2026, from https://docs.python.org/3.13/library/dataclasses.html

Tabassi, E. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.AI.100-1
