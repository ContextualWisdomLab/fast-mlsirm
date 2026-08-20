### Added

- Add a provider-neutral, source-free `ExternalValidationProfile` contract for preregistered external-validity and transportability evidence. The first slice keeps technical, construct, transportability, fairness, and decision-utility evidence distinct; preserves explicit failed/indeterminate/not-executed states; fingerprints normalized manifests; accepts provider-neutral dataset/site identities; and rejects evidence unavailable at the declared analysis cutoff. This is validation/provenance schema work only and does not move psychometric or statistical production arithmetic out of Rust.
- Reject caller-defined profile and evidence-record subclasses before reading their fields, keeping the immutable manifest boundary free of executable attribute callbacks.
- Reject overlapping development, internal-validation, and external-validation dataset identities so a transport claim cannot silently reuse a declared development cohort.
