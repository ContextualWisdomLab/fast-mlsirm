# Executed conformance provenance integrity

## Fixed

- Fail closed when a cross-engine conformance inventory contains executed `passed`, `failed`, or `indeterminate` evidence without exact run provenance.
- Require both raw-output and normalized-output SHA-256 identities for executed conformance runs while preserving optional output hashes for genuinely nonexecuted plans.
- Revalidate nested run provenance before applying the execution consistency gate so post-construction mutation cannot bypass package-owned admission.
