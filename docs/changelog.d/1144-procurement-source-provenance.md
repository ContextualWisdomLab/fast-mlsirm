# Require reconstructable procurement source identity

## Fixed

- Procurement due-diligence generation now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated or malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving release-evidence interoperability without changing psychometric or statistical numerical ownership.
