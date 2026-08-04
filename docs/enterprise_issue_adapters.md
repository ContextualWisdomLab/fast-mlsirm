# Governed enterprise issue adapters

`fast_mlsirm.scoring.enterprise_issue` is the provider-neutral evidence and
request boundary for the enterprise issue intelligence vertical. It extends the
shared rubric and scoring contracts; it does not introduce a second observation,
result, engine, calibration, ranking, or decision schema.

## Scope

The first slice provides immutable, content-addressed records for:

- enterprise source provenance;
- exact evidence spans;
- counterevidence bound to a declared claim;
- atomic issues with evidence-role separation;
- stakeholder-specific value judgments;
- candidate interventions whose causal effect is explicitly not estimated; and
- criterion-level compilation into the authoritative `ScoringRequest` contract.

The records retain fingerprints, descriptive identifiers, bounded character and
unit counts, offsets, and deeply immutable metadata. They do not retain report,
lead, complaint, customer, or prompt text. Metadata validation rejects known raw
content fields.

## Epistemic separation

Every evidence span declares exactly one `EnterpriseAssertionKind`:

| Assertion kind | Shared evidence role | Interpretation boundary |
|---|---|---|
| `direct_fact` | `supporting_evidence` | Explicitly stated in the cited source span. |
| `supported_inference` | `supporting_evidence` | Inference supported by cited evidence; not a directly stated fact. |
| `counterevidence` | `counter_evidence` | Evidence that bears against a named claim. |
| `unresolved_ambiguity` | `context_evidence` | Material uncertainty retained for review rather than imputed. |
| `stakeholder_value_judgment` | `context_evidence` | A stakeholder preference or normative judgment, not an empirical fact. |

`CounterevidenceRecord` is structurally separate from ordinary evidence spans.
`AtomicIssueRecord` rejects counterevidence placed in its supporting/context
collection. Evidence strength is therefore not silently multiplied into
materiality, and weak-evidence/high-consequence cases remain representable for a
future information-collection policy.

## Provenance and replay resistance

Factories bind records to exact normalized content and expose full SHA-256
fingerprints plus descriptive 128-bit public handles. Evidence spans also produce
the shared `EvidenceReference` required by `ScoreObservation`. The enterprise
request wrapper compiles exact issue, evidence, counterevidence, perspective, and
intervention fingerprints into one criterion-level `ScoringRequest`.

The design is consistent with the W3C PROV family principle that provenance
information should be representable and interchangeable across systems. The
repository canonicalization contract provides deterministic, hashable content;
RFC 8785 is cited as the relevant canonical-JSON standard, although the package's
existing canonicalizer is the authoritative implementation contract for this
release.

Content identity prevents accidental replay across changed source, issue,
stakeholder, or intervention revisions. It does not establish semantic truth,
source authenticity, legal admissibility, construct validity, or causal effect.

## Privacy and security boundary

The adapter stores no raw source text and allows an optional fingerprint of a
subject/customer identifier rather than the identifier itself. This is data
minimization, not anonymization. Linkability, membership inference, dictionary
attacks against low-entropy identifiers, access control, retention, deletion,
and lawful processing remain deployment responsibilities.

The NIST Privacy Framework is used as an enterprise risk-management reference;
using these contracts alone does not demonstrate conformance with that framework
or any privacy law.

## Psychometric and decision boundary

This slice performs no likelihood, gradient, Hessian, optimization, calibration,
ranking, utility, expected-value-of-information, or queue-routing arithmetic.
Future measurement must delegate to Rust-backed kernels and must preserve
criterion, judge-family, task-revision, stakeholder, and evidence provenance.
Future priority decisions must distinguish latent measurement from expected net
intervention value and must not infer causal effects from observational text
without an identified design.

No correlation, sentiment score, opaque aggregate judge score, or absence of a
review flag establishes validity, fairness, reliability, model preference, or
authorization for consequential automation.

## Minimal example

```python
from fast_mlsirm.scoring.enterprise_issue import (
    EnterpriseAssertionKind,
    build_atomic_issue_record,
    build_enterprise_source_record,
    build_evidence_span_record,
)

source = build_enterprise_source_record(
    source_id="customer_complaint",
    source_type_id="complaint_record",
    source_content_fingerprint="1" * 64,
    source_character_count=500,
    source_unit_count=20,
)
span = build_evidence_span_record(
    source=source,
    span_id="material_fact_span",
    content_fingerprint="2" * 64,
    assertion_kind=EnterpriseAssertionKind.DIRECT_FACT,
    start_offset=10,
    end_offset=40,
)
issue = build_atomic_issue_record(
    issue_id="billing_delay_issue",
    issue_family_id="service_reliability",
    domain_id="customer_operations",
    issue_content_fingerprint="3" * 64,
    issue_character_count=120,
    issue_unit_count=8,
    evidence_spans=(span,),
)
```

## References

Lebo, T., Sahoo, S., & McGuinness, D. (Eds.). (2013, April 30). *PROV-O:
The PROV ontology* (W3C Recommendation). World Wide Web Consortium.
https://www.w3.org/TR/prov-o/

National Institute of Standards and Technology. (2020). *NIST privacy
framework: A tool for improving privacy through enterprise risk management,
version 1.0*. U.S. Department of Commerce.
https://doi.org/10.6028/NIST.CSWP.01162020

Python Software Foundation. (2025). *`dataclasses`—Data classes* (Python 3.13.9
documentation). https://docs.python.org/3.13/library/dataclasses.html

Rundgren, A., Jordan, B., & Erdtman, S. (2020). *JSON canonicalization scheme
(JCS)* (RFC 8785). RFC Editor. https://doi.org/10.17487/RFC8785
