# Callback-safe inference evidence admission

## Fixed

- Seal Hessian/covariance matrix identity before NumPy materialization for second-order, covariance, and standard-error diagnostics. Exact real-numeric NumPy arrays and inert built-in square matrices remain supported; arbitrary array providers, subclasses, complex storage, and non-numeric storage fail before caller protocols or Rust dispatch.
- Validate and normalize `tol` and `rcond` as finite non-negative Rust `f64` controls before caller matrix work. Boolean, callback-bearing, non-finite, negative, and lossy controls fail closed.
- Preserve Rust ownership of positive-definiteness eigendiagnostics, inversion/pseudoinversion, and covariance-diagonal standard-error arithmetic.
