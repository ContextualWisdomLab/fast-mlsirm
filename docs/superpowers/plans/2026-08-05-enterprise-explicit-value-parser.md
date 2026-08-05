# Enterprise explicit-value parser implementation plan

## Implementation status

The bounded implementation and focused review repairs are complete. Exact-head
repository CI, security, SAST, and independent review remain the authority for
merge; this status does not expand the parser scope or claim release readiness.

## Goal

Advance issue #404 with the smallest reviewable parser slice after the governed
enterprise evidence and scoring-request contracts. Add a deterministic,
provider-neutral boundary for explicit values already present in authorized source
text. Do not perform semantic issue extraction, sentiment analysis, scoring,
calibration, ranking, utility arithmetic, causal inference, or queue routing.

## Accepted boundaries

The implementation belongs under `fast_mlsirm.scoring.enterprise_issue` and must
reuse `EnterpriseSourceRecord`, `EvidenceSpanRecord`,
`EnterpriseAssertionKind`, and the shared scoring contracts. It must not create a
parallel observation, result, engine, assessment, or rubric schema.

The parser receives source text transiently, verifies it against the exact
`EnterpriseSourceRecord` content fingerprint and character count, and returns
content-addressed records without retaining source text or clear-text customer
identifiers.

## Public contract

Add:

- `ExplicitValueKind`, with distinct values for `calendar_date`,
  `deadline_date`, `money_amount`, `frequency_count`, and
  `customer_identifier`;
- immutable `ExplicitValueRecord`, containing the exact source-record
  fingerprint, UTF-8 source-span fingerprint, Python Unicode-code-point offsets,
  kind-specific normalized payload, and parser-revision fingerprint;
- runtime-checkable `EnterpriseExplicitValueParser` protocol;
- `DeterministicExplicitValueParser`, whose parser revision is derived from its
  patterns and configured ISO 4217 alphabetic currency-code allowlist;
- `parse_enterprise_explicit_values()` as the stable convenience boundary.

`ExplicitValueRecord.to_evidence_span()` must compile the exact span into the
accepted `EvidenceSpanRecord` boundary as a directly stated fact. This means only
that the value was explicitly present in the source, not that its real-world
content is true.

## Deterministic grammar

The first implementation recognizes only deliberately narrow, auditable forms:

1. Gregorian calendar dates in extended `YYYY-MM-DD` form, validated by
   `date.fromisoformat()`;
2. deadlines expressed by an explicit marker such as `due`, `deadline`, `by`,
   or `no later than`, followed by an accepted calendar date;
3. money amounts expressed as an allowlisted three-letter ISO 4217 alphabetic
   code followed by an exact decimal amount; commas may be grouping separators,
   and values are normalized through `Decimal` constructed from text;
4. frequencies such as `3 times per month`, `4 incidents per week`, or
   `2 occurrences/year`, normalized to an integer count and one of `day`,
   `week`, `month`, `quarter`, or `year`;
5. customer/account identifiers introduced by an explicit `customer_id`,
   `customer id`, `account_id`, or `account id` label. The normalized payload
   stores only SHA-256 over the identifier token, never the token itself.

Deadline matches take precedence over the embedded calendar-date match. Other
overlapping matches are rejected or deterministically resolved by documented
kind priority; the same source occurrence must never be multiplied silently.
Results are sorted by `(start_offset, end_offset, value_kind,
explicit_value_fingerprint)` and are independent of pattern declaration order.

## Fail-closed validation

Reject:

- non-`EnterpriseSourceRecord` inputs;
- non-string source text;
- source-content fingerprint or character-count mismatch;
- syntactically date-shaped but impossible Gregorian dates;
- malformed configured currency codes;
- non-finite or signed money amounts outside the accepted grammar;
- non-positive frequency counts;
- empty or oversized customer identifiers;
- overlapping accepted records that cannot be resolved by the declared deadline
  precedence;
- record counts above the bounded parser limit;
- caller metadata containing raw/sensitive source fields.

Private normalization helpers must route non-string decimal payloads through the
same redacted `invalid_decimal_amount` assessment error used for malformed text;
they must not leak implementation-specific `AttributeError` exceptions.
Kind-specific normalized payloads must first be mappings. Non-mapping containers
and non-string frequency periods must fail through the structured
`invalid_normalized_payload` boundary rather than leaking unhashable-container or
membership-operation exceptions.

Provider callbacks run only after exact source replay validation. Their output
must be a bounded tuple of `ExplicitValueRecord` values and is rebound to the
verified source identity, source revision, code-point offsets, and exact UTF-8
span fingerprint. The stable boundary canonicalizes output ordering and rejects
forged, duplicate, out-of-bounds, or overlapping records. Non-domain callback
exceptions are redacted, and record metadata is restricted to the declared
`python_unicode_code_point` offset unit.

All public identifiers must remain descriptive, nonnumeric, and at least two
snake-case tokens. Public docstrings and statement/branch coverage for added code
must be complete.

## Tests

Add deterministic tests for:

- exact extraction of all five kinds from one mixed Unicode source;
- source slices matching returned offsets while serialized records retain no raw
  source text or customer token;
- deadline/date overlap producing one deadline record;
- parser output and fingerprints remaining invariant to configured currency-code
  input order;
- exact decimal normalization without binary-float conversion;
- invalid date, source replay, source length, protocol, payload, overlap, and
  maximum-record failures;
- custom-provider output canonicalization, exact-source and span rebinding,
  duplicate and overlap rejection, callback-error redaction, bounded output, and
  strict metadata allowlisting;
- conversion to the accepted `EvidenceSpanRecord` and shared
  `EvidenceReference` role;
- no sentiment wording effect on extracted explicit values when the explicit
  evidence is unchanged;
- 100% statement and branch coverage and complete public docstrings.

## Documentation and changelog

Extend `docs/enterprise_issue_evidence_contracts.md` with the transient-text,
code-point-offset, privacy, normalization, and interpretation boundaries. Add an
authoritative changelog fragment and keep the managed `CHANGELOG.md` block
synchronized after every fragment change; the repository parity contract remains
a required merge gate.

ISO 8601-1:2019 remains the current published basic-rules standard as of this
plan; ISO has an Edition 2 committee draft under development. ISO 4217:2015 is
the current published currency-code standard and is maintained through its
registration authority. Python regular-expression match spans are Unicode string
indices, not user-perceived grapheme-cluster positions; therefore the contract
must state its offset unit explicitly rather than imply UAX #29 grapheme
segmentation.

## References

International Organization for Standardization. (2015). *Codes for the
representation of currencies* (ISO Standard No. 4217:2015).
https://www.iso.org/standard/64758.html

International Organization for Standardization. (2019). *Date and time—
Representations for information interchange—Part 1: Basic rules* (ISO Standard
No. 8601-1:2019). https://www.iso.org/standard/70907.html

Python Software Foundation. (2025). *Python 3.13 standard library: `datetime`,
`decimal`, `hashlib`, `re`, and `typing`*. https://docs.python.org/3.13/

The Unicode Consortium. (2025). *Unicode text segmentation* (Unicode Standard
Annex No. 29, Revision 47). https://www.unicode.org/reports/tr29/
