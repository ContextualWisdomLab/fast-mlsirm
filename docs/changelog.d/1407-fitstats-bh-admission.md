# Benjamini-Hochberg evidence admission

## Fixed

- Validate the public Benjamini-Hochberg FDR control and p-value evidence before compiled-core discovery, reject callback-bearing, infinite, out-of-range, or lossy inputs without caller coercion, preserve the Rust-owned `NaN` missing-p-value contract, normalize accepted evidence losslessly through the Rust `f64` boundary, and keep historical package exports bound to the same hardened Rust-backed callable.
- Bound admitted BH evidence before value-wise or dense NumPy work: exact NumPy arrays and nested exact NumPy leaves are charged against a 20,000,000 logical-cell ceiling, while built-in list/tuple traversal has a separate 40,000,000-node budget so empty/deep fan-out cannot evade the logical envelope.