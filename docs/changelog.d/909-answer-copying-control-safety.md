# Answer-copying integer control safety

## Fixed

- Validate `n_options`, `copier`, and `source` through a shared exact-type integer marshalling boundary before answer-copying native dispatch.
- Preserve exact built-in integers and supported concrete NumPy integer scalars while rejecting booleans, integer subclasses, and arbitrary coercion providers without executing caller comparison or conversion callbacks.
- Preserve positive option-count, non-negative row-index, row-range, and distinctness checks while keeping Wollack omega, K-index, and K1/K2/S1/S2 statistical arithmetic in Rust.

Closes #909.
