# MMLE theta calculation memory optimization

## Changed

- Replaced the NumPy reference/fallback EAP expression `(posterior * nodes[None, :]).sum(axis=1)` with the algebraically equivalent matrix-vector product `posterior @ nodes`. This avoids constructing the explicit posterior-shaped broadcast product; NumPy may use optimized BLAS for matrix multiplication when available, while realized runtime remains dependent on array shape, layout, hardware, and the linked numerical library.
- Bounded the retained NumPy reference/fallback before typed conversion and large response, person-by-node, or item-by-node allocations: integer Gauss-Hermite counts are limited to NumPy's documented tested range of 1 through 100, and conservative owned-workspace estimates above 512 MiB fail closed with guidance to use the Rust backend or reduce the problem size.
