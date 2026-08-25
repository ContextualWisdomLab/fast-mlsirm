# Require reconstructable PR queue source commit provenance

## Fixed

- Make `scripts/build_pr_queue_governance.py::_source_commit()` fail closed on every unreconstructable Git outcome: non-timeout command failures and executable/OS/subprocess errors now raise a stable package-owned `RuntimeError` instead of degrading to a `source_commit: "unknown"` placeholder, while keeping the existing bounded `GIT_METADATA_TIMEOUT_SECONDS` deadline for hung `git rev-parse HEAD` children.
- Accept only canonical full lowercase SHA-1 (exactly 40 hexadecimal characters) or SHA-256 (exactly 64 hexadecimal characters) object identities as the governance source commit; empty, abbreviated, uppercase, non-hexadecimal, undersized, and oversized stdout are rejected with a stable package-owned error before any PR queue governance evidence can be emitted, so every published manifest cites a source that reconstructs the exact evidence build.

- Research basis: Ohm, Plate, Sykosch, and Meier (2020), *Backstabber's Knife Collection: A Review of Open Source Software Supply Chain Attacks*, https://doi.org/10.1007/978-3-030-52683-2_2. The methodological implication for this governance path matches the release-evidence path: mutable or unverifiable provenance identities must never be silently substituted into consumed evidence, so identity resolution fails closed at the boundary where the value is produced.
