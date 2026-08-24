# Response-time evidence admission

## Fixed

- Reject complex, object/text, callback-bearing, and arbitrary array-provider response-time evidence before real-valued marshalling or Rust-core discovery across standalone RT calibration, joint speed-accuracy calibration, and RT person-fit diagnostics, while preserving ordinary built-in real-numeric sequence and NumPy-array inputs.
- Replaced the recursive built-in-sequence walk with an explicit stack so a deeply nested response-time list/tuple (past Python's recursion limit) or a self-referential one (`a = []; a.append(a)`) rejects with a validation error instead of crashing the process with an uncaught `RecursionError` or looping forever.
