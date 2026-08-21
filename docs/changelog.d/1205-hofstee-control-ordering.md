# Validate Hofstee controls before score materialization

## Fixed

- Validate and order the four Hofstee percentage controls before caller-owned score arrays are materialized, so rejected semantic controls cannot trigger score-side array protocols before the package emits its stable validation error.
- Preserve the existing Rust-owned Hofstee ogive, intersection, fallback, and cut-score arithmetic.
