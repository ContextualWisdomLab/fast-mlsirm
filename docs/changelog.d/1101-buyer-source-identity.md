# Require reconstructable buyer-packet source identity

## Fixed

- Buyer evidence packet generation now fails closed when Git source discovery times out, fails, is unavailable, or returns an abbreviated/malformed identity instead of recording `unknown` provenance.
- Canonical full lowercase SHA-1 and SHA-256 Git object identities remain accepted, preserving interoperability without changing psychometric/statistical numerical ownership.
