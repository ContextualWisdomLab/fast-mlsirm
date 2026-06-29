## 2024-06-25 - Python 3D Array Broadcasting Bottlenecks
**Learning:** In NumPy, broadcasting arrays to 3D for pair-wise distance or gradient computations (e.g., `params.xi[:, None, :] - params.zeta[None, :, :]`) creates huge O(N*J*D) intermediate matrices that rapidly eat up memory and cause significant slowdowns as dimensions grow.
**Action:** Replace 3D broadcast calculations with vectorized dot products (e.g., `(a - b)^2 = a^2 + b^2 - 2ab`) for distance matrices. For gradients, algebraically rearrange sums to compute values using 2D matrix multiplications instead of creating 3D common term arrays.

## 2024-03-10 - Vectorize looping over dimension

**Learning:** When calculating values grouped by categorical dimensions (like item factors), calculating sums or means by iterating through dimensions `d in range(n_dims)` using boolean indexing `(items = factor_id == d)` is slow because numpy does not vectorize over the outer loop. Utilizing a 2D boolean mapping mask `(factor_id[:, None] == np.arange(n_dims))` and matrix multiplications `(@)` directly converts loop aggregations into fast C/BLAS optimized operations, yielding massive performance gains.

**Action:** Whenever noticing an outer python loop over categorical subsets (often indices or distinct labels) that aggregates numeric array data, immediately attempt to broadcast the labels into a dense or sparse 2D boolean mask and aggregate using matrix multiplication `(@)`.
