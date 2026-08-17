# S-X2 and person-fit Rust ownership fail-closed

## Fixed

- Public `s_x2()` and `person_fit()` require the compiled Rust core entrypoints and no longer fall back to Python/NumPy numerical implementations when the core or symbols are missing.
- `s_x2()` always dispatches trait `prior_mean` through the native S-X² entrypoint instead of selecting the Python reference path whenever a prior is supplied.
