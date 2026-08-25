# Benjamini-Hochberg evidence admission

## Fixed

- Validate the public Benjamini-Hochberg FDR control and p-value evidence before compiled-core discovery, reject callback-bearing or non-probability inputs without caller coercion, preserve accepted evidence losslessly through the Rust `f64` boundary, and keep historical package exports bound to the same hardened Rust-backed callable.
