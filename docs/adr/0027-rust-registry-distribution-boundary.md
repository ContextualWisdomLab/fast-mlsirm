# ADR-0027: Keep Rust implementation crates inside the `fast-mlsirm` release boundary

Status: **Accepted**  
Date: 2026-09-01  
Decision owners: fast-mlsirm maintainers  
Tracks: #1715

## Context

`crates/mlsirm-core` is the Rust numerical source of truth for `fast-mlsirm`, while `crates/fast-mlsirm-py` is its PyO3 binding crate. Both Cargo packages are implementation owners behind the Python distribution, and Cargo permits registry publication unless publication is explicitly disabled. Organization-level availability of `CARGO_REGISTRY_TOKEN` is credential availability, not product authorization.

The protected product architecture currently has one external release boundary: the `fast-mlsirm` package built through Maturin/PyO3 and published through the governed PyPI workflow. `docs/20b_product_readiness.md` keeps the sellable unit as the local Python/Rust package and says a separately versioned library or SDK is justified only after it has its own release cadence and consumers. `docs/PRD.md` permits independent Rust consumers, but does not establish independently versioned crates.io products for either internal Cargo package.

No accepted workflow publishes `mlsirm-core` or `fast-mlsirm-py` to crates.io, and neither package has a separate crates.io compatibility, ownership, checksum-verification, downstream-install, SBOM/provenance, or release-cadence contract.

## Decision

`mlsirm-core` and `fast-mlsirm-py` remain internal Rust implementation/library boundaries of the released `fast-mlsirm` product. Both Cargo manifests therefore declare `publish = false`.

Maturin/PyPI remains the public registry distribution boundary. The Rust core can continue to serve repository/workspace builds and explicitly pinned source consumers; that source reuse does not create a crates.io publication promise. The PyO3 crate remains the build manifest selected by Maturin, not a standalone Cargo-registry product.

A fleet or repository credential must never widen this boundary implicitly. There is no `cargo publish` path for either Rust implementation crate under this ADR.

## Invariants and acceptance evidence

- `crates/mlsirm-core/Cargo.toml` declares `package.publish = false`.
- `crates/fast-mlsirm-py/Cargo.toml` declares `package.publish = false`.
- `tests/test_rust_distribution_boundary.py` fails if either internal Cargo package becomes registry-publishable, if the Python package name or Maturin manifest root drifts, if the governed PyPI publisher disappears, or if a `cargo publish` command is added to that release workflow without changing this decision.
- `.github/workflows/publish-pypi.yml` remains the governed public registry publisher for the current product boundary.
- Release/security evidence treats `CARGO_REGISTRY_TOKEN` as out of scope for `fast-mlsirm` unless a successor ADR authorizes a Rust registry product.

## Consequences and trade-offs

The decision reduces accidental supply-chain exposure and prevents a broadly available organization secret from reserving or publishing an implementation crate name without an explicit product decision. It also means crates.io is not a supported installation surface for either internal Rust crate today.

Consumers that require a stable standalone Rust registry dependency cannot treat this repository as having supplied one. Creating that product later is a deliberate compatibility and release-governance project, not a credential/configuration change.

## Alternatives considered

### Publish a standalone crates.io product now

Rejected. There is no accepted crates.io release contract, independent compatibility policy, registry ownership verification, dry-run/package gate, final-publisher credential isolation, or clean downstream crates.io installation evidence.

### Leave Cargo's default publication behavior unchanged

Rejected. Absence of `publish = false` makes credential availability capable of being mistaken for publication authority and leaves the repository's release boundary ambiguous.

## Failure, recovery, and reversal

An attempted Cargo registry publication is a release-boundary failure, not a signal to remove the guard. Recovery is to keep the implementation crates unpublished and use the existing Maturin/PyPI release path.

A future standalone Rust SDK/crate may supersede this ADR only after an accepted successor defines at least: crate-name ownership, independent semantic-version/API policy, protected-head and tag identity, `cargo package --locked` and `cargo publish --dry-run --locked`, Rust/rustdoc/Clippy/security/license/coverage/recovery gates, SBOM/provenance, final-job-only registry credentials, post-publication checksum verification, and a clean downstream build from crates.io.

## Security and privacy

This decision changes no psychometric arithmetic or data handling. It narrows supply-chain authority: `CARGO_REGISTRY_TOKEN` is neither needed by nor exposed to the current build/test/package evidence path.

## Compatibility and migration

The Python/Maturin package contract is unchanged. Existing workspace and source-level Rust builds remain valid. There is no migration from a released crates.io artifact because no crates.io product is part of the accepted release surface.

## Verification

- `pytest -q tests/test_rust_distribution_boundary.py`
- normal package, Rust workspace, PyO3, security, SBOM/provenance, and release-acceptance gates on the exact integrated head
- protected-branch review and approval after the final push
