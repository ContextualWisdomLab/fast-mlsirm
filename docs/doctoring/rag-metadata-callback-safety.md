# RAG metadata callback-safety decision

## Decision

Caller-supplied RAG metadata is treated as an untrusted abstract `Mapping`.
Before package-owned provenance keys or caller allowlists are evaluated, the
mapping is materialized through the existing bounded callback-safe metadata
preflight. The resulting immutable built-in representation is then passed to
the established RAG metadata builder.

This ordering prevents an alien `__contains__`, key iterator, item iterator, or
value callback from controlling package logic before the governed exception
boundary. Failures while the mapping itself is inspected become stable,
non-reflective `AssessmentSpecError` evidence. More specific nested contract
failures, such as cyclic containers, retain their package-owned codes.

## Security rationale

The boundary addresses two distinct failure modes:

1. an uncaught caller callback can escape the public package API and destabilize
   request construction; and
2. the callback's arbitrary exception text can expose caller-controlled or
   sensitive content through error messages.

The implementation therefore does not copy the original exception message into
public evidence. It emits a fixed package-owned message while preserving the
existing allowlist, provenance, canonicalization, and no-raw-content contracts.
It adds no retrieval, scoring, LLM, statistical, or authorization behavior.

MITRE classifies unhandled exceptional conditions as CWE-248 and sensitive
details in error messages as CWE-209. The NIST Secure Software Development
Framework requires producers to identify recurring vulnerability causes and
integrate secure practices into the development lifecycle. The regression
suite uses fault-injection mappings to keep these boundary guarantees
executable.

## Verification and rollback

Verification requires all of the following:

- a valid mapping whose `__contains__` raises is accepted without invoking that
  callback;
- failed key iteration returns `invalid_rag_metadata` without reflecting caller
  text;
- nested governed errors remain specific;
- the `None` metadata path and package-managed RAG provenance remain unchanged;
- production statement and branch coverage and public docstring gates remain at
  100%.

Rollback must restore the same callback-safety and non-reflective error
properties. Do not remove the preflight merely to reduce module count, and do
not substitute broad exception text forwarding for structured package errors.

## APA 7th references

MITRE. (2026). *CWE-209: Generation of error message containing sensitive
information* (Version 4.20). Common Weakness Enumeration.
https://cwe.mitre.org/data/definitions/209.html

MITRE. (2026). *CWE-248: Uncaught exception* (Version 4.20). Common Weakness
Enumeration. https://cwe.mitre.org/data/definitions/248.html

National Institute of Standards and Technology. (2025). *Secure software
development framework (SSDF) version 1.2: Recommendations for mitigating the
risk of software vulnerabilities* (Initial public draft, NIST SP 800-218r1).
https://csrc.nist.gov/pubs/sp/800/218/r1/ipd

Python Software Foundation. (2026). *Collections abstract base classes*.
Python 3 documentation.
https://docs.python.org/3/library/collections.abc.html
