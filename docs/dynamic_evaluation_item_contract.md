# Dynamic evaluation item and run-snapshot contract

`fast_mlsirm_dynamic_evaluation_item/v1` is the domain-neutral Published
Language for an evaluation whose concrete items may be authored, sampled, or
generated at run time. It does not require a pre-existing fixed item set.

The contract separates four facts that must not be collapsed:

1. the versioned blueprint revision that governed item resolution;
2. the exact concrete item instances frozen for one run;
3. the current adjudication or validation status of each reference;
4. whether cross-version comparability has actual anchor/linking evidence.

## Cold start without fixed anchors

An `EvaluationItemSetSnapshot` may contain zero anchors. This permits pilot,
diagnostic, within-run comparison, and evidence-collection work before a stable
reference corpus exists. A zero-anchor snapshot cannot claim cross-version
linking. Its `linking_status` is either `unavailable` or `within_run_only`.

`linked` requires both:

- at least one item whose role is `anchor` and whose reference status is
  `validated`; and
- an immutable linking-evidence reference.

Using the same nominal score range, rubric label, model family, or blueprint
name does not establish comparability.

## Immutable concrete item instances

The blueprint is a plan, not the administered item set. Before evaluation, the
caller resolves the actual items and freezes them in one run snapshot. Each
item records:

- exact item-instance and blueprint-revision references;
- an opaque content reference and complete lowercase SHA-256 content digest;
- origin and evaluation role;
- reference semantics and reference status;
- exact rubric and criterion revisions;
- bounded provenance references;
- a generation-invocation reference when the origin is generated,
  perturbation, or synthetic-adversarial;
- optional seed provenance and explicit regeneration status;
- separate adjudication and validation evidence references.

The canonical contract stores no source, prompt, response, provider output, or
customer content. A content digest is provenance, not authorization, identity,
anonymity, a signature, or proof of semantic equivalence.

## Orthogonal state axes

### Origin

- `authored`
- `generated`
- `production_sample`
- `perturbation`
- `synthetic_adversarial`

### Evaluation role

- `candidate`
- `anchor`
- `challenge`
- `production_sample`

### Reference semantics

- `exact`
- `constraint`
- `acceptable_set`
- `rubric`
- `pairwise`
- `open_ended`

### Reference status

- `unresolved`
- `provisional`
- `adjudication_required`
- `adjudicated`
- `validated`
- `invalidated`

These axes are deliberately independent. For example, a generated candidate may
be adjudicated while remaining unvalidated and ineligible as an anchor.

## Adjudication is not validation

`adjudicated` requires an immutable adjudication-resolution reference. It means
that an external adjudication workflow resolved a case under its own policy. It
does not establish item fit, fairness, invariance, calibration, approval, anchor
status, or score linking.

An anchor must have `validated` reference status and at least one separate
validation-evidence reference. The hosted panel and adjudication workflow remains
owned by Psychometrics Commons. Provider-neutral observation creation remains
owned by contextual-orchestrator. This package does not mutate source
observations or perform adjudication.

## Seed and regeneration evidence

A seed, prompt revision, provider/model identity, or recorded generation input is
provenance only. It does not prove that future generation will reproduce the
same content. `regeneration_status=verified` therefore requires an independent
regeneration-evidence reference. Otherwise regeneration remains
`inputs_recorded` or `unavailable`.

Content replay and content regeneration are distinct:

- replay retrieves the exact frozen content reference/digest used by the run;
- regeneration invokes a generator again and may differ unless independently
  verified.

## DDD ownership

`fast-mlsirm` owns this reusable item/snapshot vocabulary and future Rust-owned
psychometric eligibility, calibration, fit, DIF, information, linking, and
uncertainty calculations. It does not own provider execution, item-content
storage, tenant authorization, panel assignment, adjudication workflow, hosted
persistence, or product-specific instrument activation.

- `contextual-orchestrator` creates provider-neutral generation and rater
  invocation evidence through an Anti-Corruption Layer.
- `LineageWeave` owns product-specific rubric, source-evidence, instrument, and
  provenance projections for its own product context.
- Psychometrics Commons owns hosted blueprint/run lifecycle, panel assignment,
  adjudication, persistence, authorization, and immutable result publication.
- TEPP owns temporal/event semantics and later drift or invariance monitoring.

All cross-repository consumption must use an immutable released/versioned
contract and digest. A mutable sibling PR head is development evidence, not a
production dependency.

## Fail-closed boundaries

The implementation rejects:

- malformed, normalized, control-bearing, or overlong opaque references;
- non-lowercase or incomplete SHA-256 digests;
- generated items without an exact generation invocation;
- authored or production-sampled items that claim generation authority;
- duplicate criteria, provenance references, or item-instance identities;
- mixed blueprint revisions in one run snapshot;
- adjudication references on incompatible states;
- validation evidence on unresolved/provisional/adjudication-only states;
- anchor status without validated reference evidence;
- verified regeneration without independent evidence;
- linked status without validated anchors and linking evidence;
- item sets above the bounded allocation ceiling.

No thresholding, scoring, calibration, item generation, model selection,
adjudication, or linking arithmetic occurs in this module.

## Verification

The focused contract and boundary suites use opaque placeholder references; no
example production item or fixed item bank is required. They prove zero-anchor
cold start, adjudication/validation separation, seed honesty, exact content
identity, immutable item-set resolution, resource bounds, and 100% statement and
branch coverage of the new module in the isolated verification harness.

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
