# Seal subscore scientific-evidence admission

## Fixed

- Reject caller-defined array, container, and numeric protocol providers for Haberman subscore response and item-to-subscale evidence before NumPy materialization or compiled-Rust discovery.
- Preserve exact NumPy numeric arrays and exact built-in list/tuple trees of trusted concrete Python/NumPy numeric scalars while retaining complex, shape, completeness, and subscale-domain validation.
- Keep Cronbach alpha, observed and disattenuated correlations, Haberman PRMSE, augmented-score weights, and added-value decisions in the Rust numerical owner; this change is Python validation and marshalling only.
