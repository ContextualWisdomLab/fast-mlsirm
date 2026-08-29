# ADR-0027: Governed rater published language and context ownership

Status: **Proposed**  
Date: 2026-08-29

## Context

`fast-mlsirm` already owns domain-neutral measurement contracts and numerical
interpretation, while hosted assessment lifecycle belongs to Psychometrics
Commons. ADR-0001 established that repository direction. Automated human,
model, and algorithmic rating now needs a narrower boundary that several CWL
products can consume without importing CEFR, provider-specific payloads, hosted
workflow state, or final business decisions.

The existing ecosystem contains useful but overlapping terms: model calls,
judge outputs, scores, levels, result snapshots, panel assignments, and temporal
drift. Without a canonical published language, each product can treat repeated
model calls as independent raters, discard abstentions or provider failures,
average ordinal categories, or allow a provider payload to become a placement
or certification decision. Those behaviors make rater calibration and its
failure denominator irreproducible.

Domain-Driven Design requires one ubiquitous language inside each bounded
context and an explicit translation mechanism between contexts. A shared source
repository is not justified because observation creation, numerical
calibration, hosted assessment operations, temporal monitoring, and reference
metadata have distinct authorities and release lifecycles.

## Decision

`fast-mlsirm` owns and releases the published language
`cwl_governed_rater_observation/v1`.

The published language contains:

- one `RaterInvocation` aggregate identity;
- one exact `RaterConfigurationIdentity` consisting of rater family, provider or
  employing authority, implementation revision, instruction revision,
  response-schema revision, workflow mode, and modality channel;
- one task revision, rubric revision, and response-evidence reference;
- one or more unique criterion observations;
- either an evidence-bearing ordered-category anchor or an explicit abstention;
- bounded uncertainty and review-signal references.

It deliberately excludes:

- raw responses, audio, images, and provider payloads;
- participant identity and consent;
- panel assignment and adjudication state;
- latent scores and parameter arrays;
- cut scores, levels, placement, pass/fail, certification, employment decisions,
  and other downstream policy outcomes.

The context map is:

```text
semantic-data-portal
  Measurement Context Registry / Open Host Service
        |
        v
contextual-orchestrator
  Rater Observation / Anti-Corruption Layer
        |
        v
fast-mlsirm
  Measurement Calibration / Published Language owner
        |
        v
psychometrics-commons
  Assessment Operations / customer of numerical artifacts
        |
        v
TEPP
  Temporal Measurement Monitoring / downstream analytical context
```

Cross-context integration uses released JSON Schema and language-specific SDKs
pinned by version and digest. A consumer translates its own aggregate into the
published language; it does not import another context's persistence or internal
entity types.

### Canonical serialized identity

References are opaque identity text, not display labels. Producers must send the
canonical reference exactly as intended: empty or whitespace-only references,
references longer than 256 Unicode scalar values, control characters, and
leading or trailing whitespace are rejected. Consumers must not silently trim
or otherwise normalize a reference because that would allow the same serialized
contract to acquire different identities in different SDKs.

The released JSON representation serializes `observations` as an object keyed by
canonical `criterion_ref`. The observation value therefore contains only the
status-specific observation fields; criterion identity is the object key. This
makes criterion uniqueness structural in the JSON data model rather than asking
independent SDKs to reproduce an array-wide property-uniqueness extension that
JSON Schema Draft 2020-12 does not provide. Raw JSON parsers must reject
duplicate object member names before schema validation so duplicate textual
criterion keys cannot collapse through last-value-wins parsing.

The Rust aggregate may retain observations in producer order for deterministic
in-process work, but `RaterInvocation::new` independently enforces the same
criterion uniqueness and the same published cardinality ceilings. Serialization
order is not criterion identity.

Repeated executions of one model/prompt/rubric configuration are separate
invocations nested under one configuration identity. They are not independent
rater identities. Failures and abstentions remain in the denominator through
separate events and workflow records even when they do not produce category
observations.

CEFR and any other domain vocabulary may be implemented only as a profile or
Anti-Corruption Layer over this language. No CEFR level, descriptor, linking
claim, or certification authority enters the generic contract.

## Aggregate invariants

- references are exact, non-empty, bounded, free of control characters, and
  contain no leading or trailing whitespace;
- an observed criterion has between one and 64 unique evidence references;
- an abstention has no manufactured category and an explicit reason;
- each observation has at most 32 unique review-signal references;
- an invocation contains between one and 128 criteria and no duplicate
  criterion;
- an invocation cannot represent a final score or product decision;
- contract evolution is additive only within a major version; incompatible
  changes require a new contract identifier.

## Consequences

### Benefits

- one rater identity model can serve writing, interview, portfolio, performance,
  peer-review, and future language-assessment profiles;
- numerical recovery and calibration consume a stable, domain-neutral input;
- provider APIs and product decisions remain outside the scientific contract;
- failures, abstentions, and repeated runs can be audited correctly;
- each repository preserves aggregate and transaction ownership.

### Costs

- every domain profile requires an explicit adapter;
- release ordering and digest pinning must be coordinated across repositories;
- existing CEFR-specific gateways must be migrated rather than treated as the
  generic core;
- producer SDKs must emit canonical references and a criterion-keyed observation
  object, and JSON decoders must reject duplicate member names;
- the first PR establishes contracts only; rater-calibration estimators and
  simulation evidence follow in separately reviewable slices.

## Alternatives considered

1. **Create a new rater-framework repository.** Rejected because it would own no
   independent product state or numerical authority and would become a shared
   kernel with coordinated-change coupling.
2. **Let Psychometrics Commons own all rater contracts.** Rejected because
   reusable numerical interpretation and input invariants must remain adjacent
   to the calibration engine.
3. **Keep CEFR as the core contract.** Rejected because external language-level,
   rights, linking, and claim semantics are not ubiquitous across domains.
4. **Use provider response schemas directly.** Rejected because infrastructure
   vocabulary and mutable provider behavior cannot define the measurement
   domain.
5. **Keep observations as an array and document criterion uniqueness.** Rejected
   because standard JSON Schema cannot enforce uniqueness of one property across
   otherwise different array items. Criterion-keyed serialization makes the
   invariant structural for every schema consumer.

## Verification and release gates

- Rust unit and integration tests cover aggregate rejection paths and published
  cardinality boundaries;
- Draft 2020-12 JSON Schema rejects unknown and decision fields;
- `contracts/governed-rater-observation-v1.conformance.json` is the shared
  positive/negative identity fixture for SDK conformance, including canonical
  references and duplicate textual criterion-member rejection;
- downstream consumers pin the exact release and digest;
- true-parameter recovery for severity, thresholds, discrimination,
  interactions, invocation variance, and differential rater functioning is a
  later numerical merge gate;
- CPU `f64` remains the reference implementation and GPU paths require parity;
- no CEFR profile is merged as a prerequisite for this generic contract.

## Research basis

Myford and Wolfe's two-part introduction to many-facet Rasch measurement is used
here only to ground the measurement-domain distinction between observed ratings,
rater effects, and later psychometric interpretation. It does **not** determine
the DDD repository topology or JSON encoding.

- **Part I (2003)** catalogs rater effects and provides conceptual and
  mathematical background for many-facet Rasch measurement. That supports
  preserving rater/configuration/invocation and criterion evidence as explicit
  measurement provenance rather than treating a rating as error-free truth.
- **Part II (2004)** operationally defines and demonstrates leniency/severity,
  central tendency, randomness, halo, and differential leniency/severity using
  measurement models and statistical indicators. That supports keeping the raw
  observation contract separate from later calibration and rater-effect
  diagnostics instead of embedding a final score or decision in the observation
  envelope.

## Reversal conditions

Supersede this ADR only if ownership of reusable measurement contracts and
numerical calibration is intentionally moved out of `fast-mlsirm`, with a
migration preserving independent scientific use and existing artifact digests.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Evans, E. (2003). *Domain-driven design: Tackling complexity in the heart of
software*. Addison-Wesley.

Myford, C. M., & Wolfe, E. W. (2003). Detecting and measuring rater effects
using many-facet Rasch measurement: Part I. *Journal of Applied Measurement,
4*(4), 386–422. PMID 14523257. https://pubmed.ncbi.nlm.nih.gov/14523257/

Myford, C. M., & Wolfe, E. W. (2004). Detecting and measuring rater effects
using many-facet Rasch measurement: Part II. *Journal of Applied Measurement,
5*(2), 189–227. PMID 15064538. https://pubmed.ncbi.nlm.nih.gov/15064538/

Vernon, V. (2013). *Implementing domain-driven design*. Addison-Wesley.
