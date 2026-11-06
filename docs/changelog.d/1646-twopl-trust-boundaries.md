# Compensatory 2PL trust-boundary hardening

## Fixed

- Compensatory 2PL response admission now establishes package-owned snapshots before scalar traversal or shared/native dispatch. Exact list/tuple matrices are structurally preflighted before value work, mutable rows are sealed across the whole matrix before the first scalar normalization, and exact NumPy matrices are copied into dtype-preserving package storage before dichotomous replay. The existing 20,000,000-cell envelope, minimum-item contract, 0/1/NaN semantics, and package-owned diagnostics remain unchanged.
- The Rust `fit_2pl` result is now admitted as one exact built-in result envelope before `TwoPlFit` construction. Vector outputs must have model-derived cardinalities, real finite values, a bounded likelihood trace, and lossless public `float64` representation; scalar metadata must retain exact built-in identities. Hostile result mappings/conversion protocols fail before observation, mutable result carriers are snapshotted before conversion, and public NumPy arrays are independent package-owned evidence.
- These changes affect validation, provenance sealing, resource ordering, and marshalling only. The compensatory 2PL likelihood, Gauss-Hermite/QMC/MC integration, EM/ECM estimation, latent-correlation estimation, EAP scoring, identification, and convergence arithmetic remain Rust-owned and unchanged.
