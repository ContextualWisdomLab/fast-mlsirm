# Subscore native-result binding integrity

## Fixed

- Validate the exact Rust `subscore_analysis` result mapping, scalar identities, finiteness rules, Boolean decisions, all vector/matrix cardinalities, the admitted-group subscale count, and the Rust-owned reliability/PRMSE output domains (`0 < alpha <= 1`; `0 <= PRMSE <= 1 + 1e-9`) before NumPy marshalling.
- Replay deterministic cross-field identities established by the Rust owner: `PRMSE_s == alpha`, Haberman added value is exactly `PRMSE_s > PRMSE_x`, and augmented added value is exactly `PRMSE_sx > max(PRMSE_s, PRMSE_x) + 0.01`.
- Fail closed with a stable package-owned error for stale or foreign native payloads, including callback-bearing values, malformed shapes, non-finite outputs, and internally contradictory reliability/reporting evidence; the documented `NaN` diagonal of the disattenuated-correlation matrix remains valid.
- Preserve Haberman/Sinharay PRMSE, reliability, correlation, augmented-score, and added-value arithmetic in the Rust core; Python only validates and marshals the returned evidence.
