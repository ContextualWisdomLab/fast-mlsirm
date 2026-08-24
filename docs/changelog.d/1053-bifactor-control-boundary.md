# Bifactor scoreability control trust boundary

## Fixed

- Hardened both public bifactor scoreability entry points so `general_factor` and `zero_tolerance` are validated and normalized before loading, uniqueness, or logit-slope materialization and before compiled-core discovery.
- Reject booleans, caller-defined numeric subclasses, and arbitrary conversion-protocol objects without executing their callbacks, while preserving concrete Python/NumPy scalar compatibility and Rust ownership of index/domain validation and all scoreability arithmetic.
