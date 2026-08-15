# Harden Hofstee scalar control validation

## Security

- Harden Hofstee standard-setting scalar controls so rejected booleans, scalar subclasses, arbitrary conversion providers, non-finite/out-of-range percentages, overflowed trusted integers, and inverted bound pairs fail before Rust-core discovery; genuine supported NumPy scalars remain compatible and all Hofstee numerical arithmetic remains Rust-owned.
