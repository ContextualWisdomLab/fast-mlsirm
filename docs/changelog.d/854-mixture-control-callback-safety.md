# Mixture IRT control callback safety

## Fixed

- Validate mixture-model controls before native-core discovery, accept only exact built-in or supported genuine NumPy scalar identities, preserve the Rust binding's existing model aliases and tolerance semantics, and reject hostile scalar subclasses before conversion or representation callbacks can execute.
