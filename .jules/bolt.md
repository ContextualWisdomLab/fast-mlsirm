## 2025-02-12 - einsum optimization for masked sum
**Learning:** Using element-wise multiplication `(y * observed).sum(axis=axis)` for masked arrays creates large intermediate arrays in memory.
**Action:** Use `np.einsum` with an explicitly cast float mask (e.g., `np.einsum('ij,ij->j', y, observed_float)`) to compute the masked sum without allocating full-sized intermediate arrays.
