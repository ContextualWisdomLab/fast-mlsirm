# Pin Rust 1.97.1 across verification

## Changed

- Pin local Rust builds, Python/Rust package verification, ordinary Rust tests, GPU smoke, packaging, and scheduled statistical studies to exact Rust 1.97.1 instead of a floating stable channel.
- Track the root `rust-toolchain.toml` through Dependabot so future stable compiler updates arrive as reviewable pull requests with exact-head scientific, package, GPU, security, and recovery evidence.
- Preserve the existing public crate compatibility boundary by not adding or raising `package.rust-version`; this is a repository build-baseline change, not a new downstream MSRV claim.
