# Strict artifact JSON constants

## Fixed

- Reject `NaN`, `Infinity`, and `-Infinity` by default in the shared bounded artifact JSON loader so persisted package artifacts use interoperable JSON semantics; explicit caller `parse_constant` policies remain supported.
