# Exposure-control scalar callback safety

## Fixed

- Validate CAT/exposure integer controls from exact built-in Python and genuine NumPy scalar types before caller-dispatchable coercion or Rust-core discovery, preserving integral built-in/NumPy floating controls, package-owned bounds/errors, and Rust-owned exposure, routing, scoring, posterior, recovery, and simulation arithmetic.
