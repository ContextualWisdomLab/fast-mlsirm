# Non-finite inference uncertainty preserves scientific meaning

## Fixed

- Standard errors from covariance diagonals preserve `NaN` and infinite values instead of converting them into false zero uncertainty.
- `vcov_from_hessian` rejects non-finite observed-information entries with a stable finite-entry contract.
