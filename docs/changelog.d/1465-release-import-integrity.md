# Release helper import integrity

## Fixed

- Fail closed when repository-owned bounded JSON or subprocess helpers raise an internal missing-dependency error; direct-script fallback now occurs only when the `scripts` package or the bounded helper module itself is unavailable, preserving the first causal boundary.
