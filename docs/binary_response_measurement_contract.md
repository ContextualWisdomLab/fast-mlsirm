# Binary response measurement contract

`fast_mlsirm_binary_response/v1` is the Published Language for a reusable
dichotomous-response measurement boundary. It preserves the evidence state of
each cell independently from its numeric value before a downstream Rust-owned
psychometric kernel is selected.

## Ubiquitous Language and invariants

A `BinaryResponseCell` is a Value Object. Its response state is one of
`observed`, `missing`, `not_observed`, `abstained`, `invalid`, `omitted`,
`not_applicable`, `insufficient_evidence`, or `adjudicated`.

Only `observed` and `adjudicated` cells carry a numeric value, and that value is
an exact integer `0` or `1`. Every other state carries `None`; it is never
rewritten as zero, an incorrect response, or another response state.
`adjudicated` is a value-bearing evidence state rather than a missingness state
and therefore requires a separate opaque adjudication reference. An ordinary
`observed` cell cannot claim adjudication provenance.

Opaque observation and adjudication references are exact built-in strings of
1..256 Unicode scalar values. Boundary whitespace, C0/C1 controls, and lone
UTF-16 surrogate code points are rejected instead of normalized. This keeps the
published JSON-compatible evidence serializable and prevents caller-defined
string protocols from redefining identity during admission.

A `BinaryResponseMatrix` is the Aggregate that seals a non-empty rectangular
collection of those cells. Matrix construction is bounded at 1,000,000 cells.
Only exact package-created `BinaryResponseCell` values enter the aggregate, so a
subclass cannot bypass cell admission and later inject a nonbinary value.
`responses_array()` is a marshalling boundary only: it returns a fresh float64
array where cells without a binary value are `NaN`, while zero and one remain
unchanged. The state and provenance matrices remain available separately.

No thresholding, ordinal-to-binary conversion, imputation, likelihood,
calibration, scoring, or validity decision occurs in this module.

## Context Map

The Measurement bounded context in fast-mlsirm owns the versioned binary state
vocabulary, value/matrix invariants, bounded marshalling, and reusable numerical
compute interfaces. Production psychometric arithmetic remains Rust-first.

LineageWeave is a downstream/foreign context. It owns the meaning of an
importance or evaluation item, source-evidence binding, pilot and administrator
workflow, and buyer-facing interpretation. It may translate its domain records
to this Published Language through an Anti-Corruption Layer; fast-mlsirm does
not import LineageWeave product types or reproduce its policy database.

Psychometrics Commons remains the hosted downstream product boundary and owns
persistence, participant/session lifecycle, authorization, adjudication
workflow, result publication, and deployment concerns. A content-addressed or
opaque reference in this contract is provenance, not authorization.

TEPP remains the owner of temporal-event semantics and longitudinal event
analysis. A response collected on an occasion may carry an opaque observation
reference issued by the owning workflow, but this contract does not model a
TEPP event graph or temporal ontology.

The existing rubric pilot/facets contracts remain separate because they admit
ordered polytomous categories. They must not be silently reinterpreted as this
binary Published Language. Conversely, a binary response matrix must reject any
value other than exact integer zero or one rather than thresholding it.

## Model boundary

This contract is deliberately model-neutral. Rasch/1PL, 2PL, MIRT, bifactor,
testlet, facet/rater, DIF/linking, and other estimators may consume binary
responses only through their own validated adapters and identification rules.
The existence of a valid response matrix does not establish dimensionality,
local independence, identifiability, scoreability, classification validity, or
model fit.

When a numerical adapter is added, it must preserve this state/value separation
and meet the repository's Rust-first recovery, uncertainty, convergence,
invariance, boundary, and backend-parity gates. A model that cannot represent a
downstream request without violating its assumptions must fail closed rather
than coercing the data.

## Research basis

The *Standards for Educational and Psychological Testing* require score
interpretations and testing procedures to be supported by evidence appropriate
to their intended use. This contract therefore preserves response-state and
provenance evidence before any estimator chooses a missingness or scoring
assumption; the wire contract itself does not turn nonresponse into evidence of
incorrectness.

Rose, von Davier, and Nagengast (2017) show that omitted and not-reached item
responses can have different statistical properties and develop an IRT model
that represents their response indicators explicitly. Pohl, Gräfe, and Rose
(2014) likewise evaluate omitted and not-reached responses as distinguishable
missing-response processes. Those results support retaining the observed
response and the reason it is absent as separate evidence so later model choice
can be explicit and testable. They do not imply that every missingness process
must use one particular nonignorable model.

No article text is redistributed here; the repository uses citation, DOI link,
and a narrow contract-relevant summary.

### References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.
https://www.aera.net/publications/books/standards-for-educational-psychological-testing-2014-edition

Pohl, S., Gräfe, L., & Rose, N. (2014). Dealing with omitted and not-reached
items in competence tests: Evaluating approaches accounting for missing
responses in item response theory models. *Educational and Psychological
Measurement, 74*(3), 423–452. https://doi.org/10.1177/0013164413504926

Rose, N., von Davier, M., & Nagengast, B. (2017). Modeling omitted and
not-reached items in IRT models. *Psychometrika, 82*(3), 795–819.
https://doi.org/10.1007/s11336-016-9544-7
