# Criterion-bound dynamic evaluation contract

`fast_mlsirm_dynamic_evaluation_item/v1` is the domain-neutral Published
Language for evaluations whose concrete items may be authored, sampled,
perturbed, or generated at run time.

A fixed item bank is optional. **An explicit evaluation criterion set is not.**
No item or run is admitted until the intended use, construct, scope, rubric,
criterion definitions, evidence rules, response semantics, response categories,
and abstention behavior have been frozen under one immutable criterion-set
revision.

The contract therefore separates five facts that must not be collapsed:

1. what construct and intended use govern the evaluation;
2. what evidence and response rules define each criterion;
3. what concrete item instances were administered in one run;
4. what adjudication or validation state applies to an item reference; and
5. whether cross-version comparability has actual anchor and linking evidence.

## Criteria precede evaluation

An `EvaluationCriterionSetSnapshot` is required before a dynamic item or run can
be built. It contains a non-empty set of `EvaluationCriterionDefinition` values
and binds them to the exact:

- criterion-set and criterion revision identities;
- blueprint and rubric revisions;
- intended-use and construct references;
- population, language, and domain scopes.

Every criterion is content-addressed. It carries an immutable reference and
SHA-256 digest for:

- the criterion definition;
- the admissible-evidence rule;
- the evidence-exclusion rule;
- the response semantics;
- the abstention rule;
- the not-observable rule; and
- every admissible response-category definition.

A category may be unordered or may use a complete contiguous zero-based order.
Partial, duplicate, or gapped category ordering fails closed. References alone
do not establish meaning; their immutable artifact digests are part of the
criterion fingerprint.

A criterion can be proposed dynamically, but a proposal is not executable
measurement policy. It must first be reviewed and published as a new immutable
criterion-set revision. Changing a criterion, evidence rule, response category,
or scope creates a new revision; it does not mutate an existing run.

## Dynamic items under fixed meaning

The blueprint is a plan, not the administered item set. Before evaluation, the
caller resolves the actual items and freezes them in one run snapshot. Each
item records:

- exact item-instance and blueprint-revision references;
- an opaque content reference and complete lowercase SHA-256 content digest;
- origin and evaluation role;
- reference semantics and reference status;
- exact rubric revision;
- the exact criterion-set identity and digest;
- one or more criterion references that exist in that set;
- bounded provenance references;
- a generation-invocation reference for generated, perturbed, or synthetic
  adversarial items;
- optional seed provenance and explicit regeneration status; and
- separate adjudication and validation evidence references.

The item builder rejects an unregistered criterion, a foreign blueprint, a
foreign rubric, a missing criterion set, or a mutated criterion artifact.

A run binds the same criterion-set identity and digest. Every item must retain
that binding, and every criterion declared by the criterion set must be
administered by at least one item. A run therefore cannot silently omit a
criterion or substitute a different rubric after item generation.

`EvaluationItemSetSnapshot.items` is mathematical membership, not administration
sequence. Exact admitted items are canonicalized by immutable
`item_instance_ref` before the run fingerprint is calculated, so two callers
cannot create different snapshot identities by supplying the same set in a
different container order. Administration sequence and event time are separate
evidence and remain outside this set contract; TEPP owns temporal/event
composition.

The canonical contract stores no source, prompt, response, provider output, or
customer content. A content digest is provenance, not authorization, anonymity,
a signature, semantic equivalence, or permission to disclose source material.

## Cold start without fixed anchors

An `EvaluationItemSetSnapshot` may contain zero anchors. This permits pilot,
diagnostic, within-run comparison, and evidence collection before a stable
reference corpus exists.

A zero-anchor snapshot cannot claim cross-version linking. Its `linking_status`
is either `unavailable` or `within_run_only`.

`linked` requires both:

- at least one item whose role is `anchor` and whose reference status is
  `validated`; and
- an immutable linking-evidence reference.

Using the same nominal scale, rubric label, model family, criterion name, or
blueprint identifier does not establish comparability.

## Orthogonal state axes

Item origin, operational role, reference semantics, reference status,
regeneration evidence, adjudication provenance, validation evidence, and
linking status remain independent.

For example, a generated candidate may be adjudicated while remaining
unvalidated and ineligible as an anchor. An adjudication resolution does not
establish item fit, calibration, fairness, invariance, approval, anchor status,
or score linking.

## Seed and regeneration evidence

A seed, prompt revision, provider/model identity, or recorded generation input is
provenance only. It does not prove that future generation will reproduce the
same content. `regeneration_status=verified` therefore requires an independent
regeneration-evidence reference.

Content replay and regeneration are distinct:

- replay retrieves the exact frozen content reference and digest used by the
  run;
- regeneration invokes a generator again and may differ unless independently
  verified.

## DDD ownership

`fast-mlsirm` owns this reusable criterion/item/run vocabulary and future
Rust-owned psychometric eligibility, calibration, fit, DIF, information,
linking, and uncertainty calculations. It does not own provider execution,
item-content storage, tenant authorization, panel assignment, adjudication
workflow, hosted persistence, or product-specific instrument activation.

- `contextual-orchestrator` creates provider-neutral generation and rater
  invocation evidence through an Anti-Corruption Layer, using the exact
  criterion-set identity and digest supplied by its caller.
- `LineageWeave` owns product-specific criterion, rubric, source-evidence,
  instrument, and provenance projections for its own bounded context.
- Psychometrics Commons owns hosted blueprint/run lifecycle, panel assignment,
  adjudication, persistence, authorization, and immutable result publication.
- TEPP owns temporal/event semantics and later drift or invariance monitoring.

All cross-repository consumption must use an immutable released/versioned
contract and digest. A mutable sibling PR head is development evidence, not a
production dependency.

## Fail-closed boundaries

The implementation rejects:

- a missing or empty criterion set;
- criterion or response-category definitions without complete SHA-256 bindings;
- duplicate criterion, criterion-revision, category, or category-order identity;
- partial or non-contiguous ordered response categories;
- item criteria that are not registered in the frozen criterion set;
- criterion-set, item, rubric, or blueprint substitution;
- runs that do not administer every declared criterion;
- malformed, normalized, control-bearing, or overlong opaque references;
- generated items without an exact generation invocation;
- authored or production-sampled items that claim generation authority;
- duplicate provenance references or item-instance identities;
- adjudication references on incompatible states;
- validation evidence on unresolved/provisional/adjudication-only states;
- anchor status without validated reference evidence;
- verified regeneration without independent evidence;
- linked status without validated anchors and linking evidence; and
- criterion, category, reference, or item collections above bounded ceilings.

No thresholding, scoring, calibration, item generation, model selection,
adjudication, or linking arithmetic occurs in this package.

## Verification

The focused suites use synthetic, content-addressed criterion artifacts and
opaque item references. No production item bank or fixed anchor example is
invented. The tests cover criterion/category integrity, zero-anchor cold start,
adjudication/validation separation, seed honesty, exact content identity,
criterion coverage, canonical item-set identity, immutable item/run resolution,
and bounded hostile inputs.

Repository-hosted exact-head CI, security, package, coverage, and independent
review remain authoritative before integration.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item
generation. *International Journal of Testing, 12*(3), 273–298.
https://doi.org/10.1080/15305058.2011.635830

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). A brief introduction to
evidence-centered design. *ETS Research Report Series, 2003*(1), i–29.
https://doi.org/10.1002/j.2333-8504.2003.tb01908.x
