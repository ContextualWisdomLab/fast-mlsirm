# Rust registry distribution boundary doctoring

Status: governed by [ADR-0027](../adr/0027-rust-registry-distribution-boundary.md)  
Owner: `fast-mlsirm` release/package boundary

## Expected product shape

The released product is `fast-mlsirm`, built with Maturin/PyO3 and published through the governed PyPI workflow. `crates/mlsirm-core` is the Rust numerical source of truth behind that package; it is not a separately published crates.io product.

`docs/PRD.md` permits independent Rust consumers, but that reuse does not imply a crates.io publication contract. A separately published Rust SDK requires its own accepted ADR, compatibility policy, release cadence, registry ownership, release evidence and consumers.

## Doctoring checks

Run:

```bash
pytest -q tests/test_rust_distribution_boundary.py
```

The contract requires all of the following:

- `crates/mlsirm-core/Cargo.toml` has `publish = false`;
- `pyproject.toml` still publishes the `fast-mlsirm` product and binds Maturin to `crates/fast-mlsirm-py/Cargo.toml`;
- `.github/workflows/publish-pypi.yml` still contains the governed PyPI publisher; and
- the current public package workflow contains no `cargo publish` command.

A failure is a release-boundary defect. Do not repair it by deleting the guard merely because `CARGO_REGISTRY_TOKEN` is available. Credential availability is not product authority.

## If a standalone Rust crate is proposed later

Keep `publish = false` until an accepted successor ADR and implementation provide crate-name ownership verification, protected release identity, semantic-version/API compatibility policy, package/dry-run gates, Rust/rustdoc/Clippy/security/license/coverage/recovery evidence, SBOM/provenance, final-job-only registry credential exposure, published checksum verification and a clean downstream crates.io build.
