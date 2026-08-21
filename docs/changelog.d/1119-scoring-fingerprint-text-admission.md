# Scoring fingerprint text admission

## Fixed

- Require caller-supplied SHA-256 scoring provenance to be an exact built-in string before validation or retention, preventing valid-looking string subclasses from crossing the package trust boundary as canonical fingerprints.
- Apply the same exact built-in text boundary to structured scoring error code, path, and message fields.
- Reject caller-defined scalar subclasses in bounded scoring metadata before canonicalization or digesting.
