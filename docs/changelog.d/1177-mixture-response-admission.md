# Mixture-response admission hardening

## Fixed

- Reject complex mixture-IRT response evidence before real-valued narrowing so caller data cannot silently project onto a different observed 0/1 pattern before Rust validation.
- Reject object-dtype response storage before per-element numeric coercion, including Python complex objects and caller-defined conversion callbacks, with the package-owned real-valued input error.
- Reject positive and negative infinity instead of treating them as undocumented missing responses, while preserving `NaN` as the documented MAR missingness representation.
- Keep mixture likelihood, posterior, EM updates, restart selection, canonical class ordering, convergence, and EAP arithmetic unchanged in the Rust numerical core.
