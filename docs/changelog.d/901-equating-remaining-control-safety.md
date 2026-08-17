# Harden remaining equating controls before native discovery

## Changed

- Validate circle-arc method/point/scalar controls, nominal-weights score ceilings and synthetic-population weight, and the composite-linking exponent before compiled-core discovery.
- Reject caller-defined scalar/container subclasses and arbitrary conversion providers without executing their conversion, comparison, representation, hashing, or iteration callbacks.
- Preserve exact built-in and genuine NumPy scalar compatibility while keeping circle-arc geometry, nominal-weights moments, composite-linking weight arithmetic, and all result-affecting equating mathematics in Rust.
