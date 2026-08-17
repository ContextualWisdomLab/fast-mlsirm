# Cognitive-diagnosis native control boundary

## Security

- Validate CDM-family `max_iter`, `tol`, and DINA/DINO model selectors before compiled-core discovery. Only exact built-in values and explicitly supported concrete NumPy scalar types are normalized; booleans, subclasses, protocol providers, non-finite/out-of-range values, and unknown model selectors fail locally without executing caller conversion callbacks. Rust-owned psychometric arithmetic and result schemas are unchanged.
