# Require reconstructable PR queue source identity

## Fixed

- PR queue governance evidence now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated or malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving governance-evidence interoperability without changing psychometric or statistical numerical ownership.
