# MMLE theta calculation memory optimization

## Changed

- Replaced the NumPy reference/fallback EAP expression `(posterior * nodes[None, :]).sum(axis=1)` with the algebraically equivalent matrix-vector product `posterior @ nodes`. This avoids constructing the explicit posterior-shaped broadcast product; NumPy may use optimized BLAS for matrix multiplication when available, while realized runtime remains dependent on array shape, layout, hardware, and the linked numerical library.
