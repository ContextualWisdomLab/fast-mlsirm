# Subscore native-result binding integrity

## Fixed

- Validate the exact Rust `subscore_analysis` result mapping, scalar identities, finiteness rules, Boolean decisions, and all vector/matrix cardinalities before NumPy marshalling.
- Fail closed with a stable package-owned error for stale or foreign native payloads, including callback-bearing values, malformed shapes, and non-finite outputs; the documented `NaN` diagonal of the disattenuated-correlation matrix remains valid.
- Preserve Haberman/Sinharay PRMSE, reliability, correlation, augmented-score, and added-value arithmetic in the Rust core; Python only validates and marshals the returned evidence.
