# Bounded mixed item-model validation

## Security

- Bounded `item_models` iterable consumption to at most one look-ahead entry beyond the calibrated item count, rejected arbitrary non-string model controls before caller `__str__`/`__repr__` hooks can execute, and removed rejected model content from public validation errors while preserving accepted aliases and Rust-owned calibration numerics.
