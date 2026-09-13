# Subscore native-result binding integrity

## Fixed

- Validate the exact Rust `subscore_analysis` result mapping, scalar identities, finiteness rules, Boolean decisions, all vector/matrix cardinalities, the admitted-group subscale count, and the Rust-owned reliability/PRMSE output domains (`0 < alpha <= 1`; `0 <= PRMSE <= 1 + 1e-9`) before NumPy marshalling.
- Replay deterministic matrix structure established by the Rust owner: both the observed `(K+1) x (K+1)` correlation matrix and the off-diagonal `K x K` disattenuated-correlation matrix must remain exactly symmetric, while the documented disattenuated diagonal remains `NaN`.
- Replay deterministic cross-field identities established by the Rust owner: `PRMSE_s == alpha`, Haberman added value is exactly `PRMSE_s > PRMSE_x`, and augmented added value is exactly `PRMSE_sx > max(PRMSE_s, PRMSE_x) + 0.01`.
- Fail closed with a stable package-owned error for stale or foreign native payloads, including callback-bearing values, malformed shapes, non-finite outputs, asymmetric correlation evidence, and internally contradictory reliability/reporting evidence.
- Preserve Haberman/Sinharay PRMSE, reliability, correlation, augmented-score, and added-value arithmetic in the Rust core; Python only validates and marshals the returned evidence.
