# Improve Python fitstats matrix allocations

## Changed

- Replaced the large NumPy `(v * observed).sum(axis=0)` float allocation in Python `infit` fallback logic with a bounded mixed-dtype `np.sum(v, axis=0, where=observed)` reduction.
- Switched the memory-intensive `(resid2 / v * observed)` `outfit` reduction to an in-place `np.divide` and unmasked sum, since `resid2` already contains the missing-data mask.
