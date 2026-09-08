# Rust 1.98.1 repository compiler baseline

## Changed

- Raised the reviewed repository build and scientific-study compiler baseline from Rust 1.97.1 to 1.98.1 while leaving published crate MSRV metadata unchanged.
- Bound every pinned CI and statistical-study `dtolnay/rust-toolchain` input to the exact compiler channel declared by `rust-toolchain.toml`, preventing a Dependabot manifest update from silently leaving workflow setup and reproducibility evidence on the predecessor compiler.
- Compiler-baseline pull requests that change `rust-toolchain.toml` now run the repository's release-mode statistical-study workflow on the pull-request head, so ignored-study execution, literature recovery, GRM Monte Carlo recovery, and GPU-versus-CPU recovery parity are current-head evidence rather than schedule/tag-only evidence.
- A synchronized compiler-baseline pull-request head now cancels an obsolete same-ref statistical-study run instead of serializing current-head Monte Carlo evidence behind the predecessor head. Scheduled/tag runs keep their own refs, while PR evidence converges on the latest exact source head without sample reduction or study omission.
