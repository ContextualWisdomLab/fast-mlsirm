# Serving bundle export requires Rust core

## Fixed

- `export_serving_bundle` fails closed when the compiled Rust core is unavailable
  instead of shipping incomplete bundles with null `eapsum_tables`.
