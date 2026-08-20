# Production backend boundary

## Changed

- Restrict production `FitConfig` and CLI backend selection to Rust (`rust` or
  fail-closed `auto`). Move the NumPy parity fit behind the explicit
  `fast_mlsirm.fit_reference` API and `fit --reference` mode, preserving
  testable parity without allowing an implicit production owner switch.
