# Require reconstructable benchmark source identity

## Fixed

- Benchmark evidence generation now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated/malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving repository interoperability without changing psychometric/statistical numerical ownership.
