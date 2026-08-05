# Enterprise semantic issue provider boundary

## Added

- Added runtime-checkable `EnterpriseAtomicIssueExtractor` and
  `extract_enterprise_atomic_issues` as a provider-neutral, provider-SDK-free trust
  boundary that returns the existing canonical `AtomicIssueRecord` contract.
- Added bounded exact source-packet replay, UTF-8 and Python code-point span
  verification, fresh nested issue/evidence/counterevidence reconstruction,
  deterministic ordering, duplicate and overlap rejection, and redacted provider
  failures without retaining raw enterprise text.
- Added `StaticEnterpriseIssueExtractor` as an offline fixture and integration
  adapter that performs no NLP, sentiment analysis, issue discovery, scoring,
  ranking, utility, or causal arithmetic.
- Added deterministic order-invariance, all-assertion-kind preservation,
  malicious provider, source mutation, span replay, subclass, privacy, prolific
  collection, duplicate identity, overlap, and complete statement/branch coverage
  tests for the next issue #404 workflow slice.
