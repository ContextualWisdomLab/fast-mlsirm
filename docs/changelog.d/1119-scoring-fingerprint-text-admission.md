# Scoring fingerprint text admission

## Fixed

- Require caller-supplied SHA-256 scoring provenance to be an exact built-in string before validation or retention, preventing valid-looking string subclasses from crossing the package trust boundary as canonical fingerprints.
