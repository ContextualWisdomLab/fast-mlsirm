# CI pytest fail-closed gates

## Fixed

- Hash-lock Hypothesis in the Python CI requirements so
  `tests/test_fuzz_properties.py` runs instead of collection-skipping.
- Record the `STAGE5_HIGH_Q` study-grid skips and the no-adapter GPU parity
  skip on the default pytest allowlist. The GPU smoke job still executes
  that parity test on lavapipe and rejects a skip.
- Append deprecated DIF docstrings from the function objects directly, and
  import public API inventory modules only when the name is a `fast_mlsirm`
  dotted identifier.
