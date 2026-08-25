# Callback-safe, bounded, and lossless inference evidence admission

## Fixed

- Seal Hessian/covariance matrix identity before NumPy materialization for second-order, covariance, and standard-error diagnostics. Exact real-numeric NumPy arrays and inert built-in square matrices remain supported; arbitrary array providers, subclasses, complex storage, and non-numeric storage fail before caller protocols or Rust dispatch.
- Validate and normalize `tol` and `rcond` as finite non-negative Rust `f64` controls before caller matrix work. Boolean, callback-bearing, non-finite, negative, and lossy controls fail closed.
- Apply a 20,000,000-logical-cell ceiling to trusted square Hessian/covariance evidence before dense `float64` materialization. Exact NumPy matrices are charged from inert shape metadata, and built-in square dimensions are bounded before row replay, preventing zero-allocation broadcast views or oversized built-in matrices from triggering unbounded dense allocation.
- Require every admitted matrix entry to preserve its numeric identity through Rust `f64` normalization. Built-in and concrete NumPy integers or wider floating values that would silently round during binary64 conversion fail before native inference work; exactly representable values and the existing non-finite covariance-diagonal semantics remain supported.
- Preserve Rust ownership of positive-definiteness eigendiagnostics, inversion/pseudoinversion, and covariance-diagonal standard-error arithmetic.