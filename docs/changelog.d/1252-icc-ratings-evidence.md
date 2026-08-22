# ICC ratings evidence admission

## Fixed

- Preserve callback-free ICC semantic controls while also rejecting callback-bearing, complex, Boolean, or non-real ratings before native discovery; trusted numeric arrays and built-in numeric sequences still marshal to the unchanged Rust ICC implementation.
- Preserve the established Boolean-rating diagnostic for trusted built-in/NumPy-Boolean sequences and the actionable `NaN` missingness guidance for NumPy `MaskedArray` subclasses without reopening caller-defined array or scalar callbacks.
