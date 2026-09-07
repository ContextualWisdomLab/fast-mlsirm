# Rust 1.98.1 repository compiler baseline

## Changed

- Raised the reviewed repository build and scientific-study compiler baseline from Rust 1.97.1 to 1.98.1 while leaving published crate MSRV metadata unchanged.
- Bound every pinned CI and statistical-study `dtolnay/rust-toolchain` input to the exact compiler channel declared by `rust-toolchain.toml`, preventing a Dependabot manifest update from silently leaving workflow setup and reproducibility evidence on the predecessor compiler.
