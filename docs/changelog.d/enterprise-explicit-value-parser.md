# Deterministic enterprise explicit-value parser

## Added

- Added a provider-neutral deterministic parser for verified explicit calendar
  dates, marked deadlines, allowlisted currency amounts, recurrence frequencies,
  and labeled customer or account identifiers under
  `fast_mlsirm.scoring.enterprise_issue`.
- Added exact source fingerprint and Python Unicode-code-point-count replay,
  UTF-8 span fingerprints, deterministic deadline precedence, overlap rejection,
  bounded output, and hashed identifier payloads without retaining source text or
  clear-text customer identifiers.
- Added fail-closed custom-parser output rebinding, callback redaction, exact span
  verification, duplicate/overlap rejection, and strict explicit-value metadata.
  Provider-owned records are reconstructed as fresh canonical instances,
  manually supplied offsets share the enterprise source-character bound, and
  deterministic candidate producers stop at the configured limit plus one rather
  than exhausting unexpectedly prolific iterators.
- Added immutable content-addressed records that compile exact occurrences into
  the existing directly stated `EvidenceSpanRecord` boundary without adding a
  parallel scoring, observation, result, or engine schema.
- Added deterministic, protocol, privacy, security, exact-decimal, offset,
  ordering-invariance, metamorphic sentiment-independence, and fail-closed tests,
  plus APA 7th standards traceability and conservative interpretation limits.
