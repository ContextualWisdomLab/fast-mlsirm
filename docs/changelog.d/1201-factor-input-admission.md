# Factor input admission hardening

## Fixed

- Reject complex and non-real-numeric factor-analysis, reliability, and Velicer MAP evidence before real-valued marshalling can alter caller data or execute object-element conversion.
- Normalize trusted `n_factors` and `max_m` integer controls before caller array materialization and Rust-core discovery while preserving concrete NumPy integer compatibility.
