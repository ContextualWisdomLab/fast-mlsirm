# Continuous-response-model control callback safety

## Fixed

- Validate CRM quadrature, iteration, and tolerance controls before native-core discovery; accept only exact built-in or supported genuine NumPy scalar identities; preserve the Rust quadrature domain and convergence tolerance contract; and reject hostile scalar subclasses before caller-controlled conversion, comparison, ufunc, or representation callbacks can execute.
