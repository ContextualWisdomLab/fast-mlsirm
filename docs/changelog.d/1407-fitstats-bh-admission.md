# Benjamini-Hochberg evidence admission

## Fixed

- Validate the public Benjamini-Hochberg FDR control and p-value evidence before compiled-core discovery, reject callback-bearing, infinite, out-of-range, or lossy inputs without caller coercion, preserve the Rust-owned `NaN` missing-p-value contract, normalize accepted evidence losslessly through the Rust `f64` boundary, and keep historical package exports bound to the same hardened Rust-backed callable.
