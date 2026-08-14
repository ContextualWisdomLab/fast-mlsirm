# RAG metadata callback-safety decision

## Decision

Caller-supplied RAG metadata is treated as an untrusted abstract `Mapping`.
The public RAG request contract currently permits exactly one caller field,
`evaluation_split`, and that field is a descriptive identifier rather than an
arbitrary JSON subtree.

The package therefore enumerates and validates caller keys exactly once,
rejects package-managed, duplicate, malformed, or unsupported keys before any
value is requested, and reads each authorized value exactly once. String
subclasses are converted through the built-in `str` implementation before the
existing identifier validator runs. The resulting built-in mapping is then
passed to the established RAG provenance builder.

This ordering prevents an alien `__contains__`, repeated key iterator, or value
callback from controlling package logic before the governed exception boundary.
A mapping cannot present an allowed key set during authorization and a different
key set during value capture. Invalid non-string `evaluation_split` values are
rejected as identifiers without traversing them as nested metadata.

## Security rationale

The boundary addresses three distinct failure modes:

1. an uncaught caller callback can escape the public package API and destabilize
   request construction;
2. the callback's arbitrary exception text can expose caller-controlled or
   sensitive content through error messages; and
3. re-enumerating a mutable or adversarial mapping after authorization creates a
   check/use race in which unapproved keys or values can appear after the
   allowlist decision.

The implementation therefore does not copy callback exception messages into
public evidence, does not call mapping membership callbacks, does not
re-enumerate caller keys after authorization, and does not broaden the metadata
schema merely to make defensive copying generic. It preserves the existing
allowlist, provenance, canonicalization, identifier, and no-raw-content
contracts. It adds no retrieval, scoring, LLM, statistical, or authorization
behavior.

MITRE classifies unhandled exceptional conditions as CWE-248, sensitive details
in error messages as CWE-209, and check/use races as CWE-367. The current final
NIST Secure Software Development Framework authority is SP 800-218 version 1.1;
it requires secure practices to be integrated into the software lifecycle and
emphasizes reducing recurrence by addressing vulnerability root causes. The
regression suite uses fault-injection mappings to keep these boundary guarantees
executable.

NIST SP 800-218 Rev. 1 / SSDF version 1.2 remains an initial public draft at
this decision date. It is tracked as prospective guidance, not used as the
normative release authority for this bounded change.

## Verification and rollback

Verification requires all of the following:

- a valid mapping whose `__contains__` raises is accepted without invoking that
  callback;
- authorized keys are enumerated once and each authorized value is read once;
- package-managed and unsupported keys fail before value callbacks;
- first-step, late-iteration, and value callback failures become stable,
  non-reflective package errors;
- hostile string subclasses are normalized without invoking their callbacks;
- invalid non-string split values fail as `invalid_evaluation_split` without
  nested traversal;
- the `None` metadata path and package-managed RAG provenance remain unchanged;
- production statement and branch coverage and public docstring gates remain at
  100%.

Rollback must restore the same callback-safety, single-pass authorization, and
non-reflective error properties. Do not remove the preflight, re-enumerate
caller keys after authorization, forward callback exception text, or broaden
`evaluation_split` into arbitrary nested metadata.

## APA 7th references

MITRE. (2026). *CWE-209: Generation of error message containing sensitive
information* (Version 4.20). Common Weakness Enumeration.
https://cwe.mitre.org/data/definitions/209.html

MITRE. (2026). *CWE-248: Uncaught exception* (Version 4.20). Common Weakness
Enumeration. https://cwe.mitre.org/data/definitions/248.html

MITRE. (2026). *CWE-367: Time-of-check time-of-use (TOCTOU) race condition*
(Version 4.20). Common Weakness Enumeration.
https://cwe.mitre.org/data/definitions/367.html

National Institute of Standards and Technology. (2022). *Secure software
development framework (SSDF) version 1.1: Recommendations for mitigating the
risk of software vulnerabilities* (NIST SP 800-218).
https://doi.org/10.6028/NIST.SP.800-218

National Institute of Standards and Technology. (2025). *Secure software
development framework (SSDF) version 1.2: Recommendations for mitigating the
risk of software vulnerabilities* (Initial public draft, NIST SP 800-218 Rev.
1). https://csrc.nist.gov/pubs/sp/800/218/r1/ipd

Python Software Foundation. (2026). *Collections abstract base classes*.
Python 3 documentation.
https://docs.python.org/3/library/collections.abc.html
