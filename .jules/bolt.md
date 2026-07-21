## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** In highly mathematical Python operations like those in `fast_mlsirm/objective.py` and `math.py`, intermediate array allocation and boolean indexing (e.g. `x[x >= 0]`) act as significant performance bottlenecks because they copy memory. Similarly, using `np.sum(x ** 2, axis=1)` creates an intermediate squared array.
**Action:** Use advanced numpy vectorization to skip allocations. Replace boolean masking and element-wise assignment with functions like `np.clip` or `np.logaddexp` which are highly optimized in C. Replace reductions over intermediate arrays with `np.einsum` (e.g., `np.einsum('ij,ij->i', xi, xi)`) which skips the intermediate allocation.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** `np.sum(x * x)` 패턴은 파이썬 내에서 곱셈을 위한 새로운 중간 배열을 메모리에 할당하고 이후에 그 배열의 합을 구하게 되어 성능 저하를 야기합니다. `np.vdot(x, x)` (또는 `np.einsum`)를 활용하면 중간 메모리 할당을 우회할 수 있어 속도가 비약적으로 증가합니다.
**Action:** 거대한 배열의 크기나 요소 수와 관련된 최적화 시, `np.sum(x * x)` 대신 `np.vdot(x, x)`를 사용해 오버헤드를 방지합니다.

## 2024-07-08 - Euclidean distance optimizations
**Learning:** Using `np.sqrt(np.sum((xi[:, None, :] - zeta[None, :, :])**2, axis=2))` to compute pairwise Euclidean distances creates a massive intermediate 3D array of shape (N, J, D), which becomes a major performance and memory bottleneck for large arrays.
**Action:** Replace 3D broadcasting with optimized 2D dot products. Compute squared norms individually (`(xi * xi).sum(axis=1)[:, None]`, etc.) and use `np.sqrt(np.maximum(sq_xi - 2 * np.dot(xi, zeta.T) + sq_zeta, 0.0))` to compute distance while keeping memory complexity O(N*J) and using BLAS-optimized dot products.

## 2024-03-10 - Vectorize looping over dimension

**Learning:** When calculating values grouped by categorical dimensions (like item factors), calculating sums or means by iterating through dimensions `d in range(n_dims)` using boolean indexing `(items = factor_id == d)` is slow because numpy does not vectorize over the outer loop. Utilizing a 2D boolean mapping mask `(factor_id[:, None] == np.arange(n_dims))` and matrix multiplications `(@)` directly converts loop aggregations into fast C/BLAS optimized operations, yielding massive performance gains.

**Action:** Whenever noticing an outer python loop over categorical subsets (often indices or distinct labels) that aggregates numeric array data, immediately attempt to broadcast the labels into a dense or sparse 2D boolean mask and aggregate using matrix multiplication `(@)`.

## 2025-05-18 - Replacing boolean `.sum() == 0` with `~.any()`
**Learning:** Checking for the absence of truthy values in boolean arrays using `.sum(axis=...) == 0` causes numpy to allocate a new integer array, which is inefficient.
**Action:** Replace `boolean_array.sum(axis=...) == 0` with `~boolean_array.any(axis=...)` to skip integer allocation and perform the check much faster.

## 2025-05-19 - Intermediate allocations in distance calculations
**Learning:** `(true_xi * true_xi).sum(axis=1)` in Euclidean distance formulas creates an unnecessary intermediate 2D array before performing the sum over the axis. This can cause performance bottlenecks across many function calls.
**Action:** Replace `(x * x).sum(axis=1)` with `np.einsum('ij,ij->i', x, x)` when computing pairwise Euclidean distances to avoid allocating the intermediate 2D array and achieve measurable performance gains.
## 2025-05-19 - Vectorized intermediate allocations during gradients
**Learning:** Operations like `(e * a[None, :] * theta).sum(axis=0)` and `grad_theta = (e * a[None, :]) @ idx` create full-sized N x J intermediate arrays. For larger matrices, this increases memory allocation time significantly.
**Action:** Always factor out values from sums over axes or embed operations in pre-existing broadcast arrays. For example, replace `(e * a[None, :] * theta).sum(axis=0)` with `(e * theta).sum(axis=0) * a` and replace `(e * a[None, :]) @ idx` with embedding `a` into the indicator variable `idx[np.arange(e.shape[1]), factors] = a` directly so that `e @ idx` avoids building an intermediate N x J array.

## 2025-05-19 - Fast reduction of boolean masks over 3D arrays
**Learning:** Broadcasting a 2D boolean mask `observed[:, :, None]` and performing element-wise multiplication with a 3D array (`onehot` or `prob`) followed by `.sum(axis=0)` creates a massive intermediate array of shape `(N, J, C)`. For large data sizes (e.g. `N=5000, J=100, C=5`), this memory allocation and copying dominates execution time. Furthermore, using `np.einsum` with boolean arrays directly is slow due to numpy's internal handling of boolean inputs in `einsum`.
**Action:** When aggregating 3D data masked by a 2D boolean array across an axis, explicitly cast the boolean mask to the target numeric type (`observed.astype(prob.dtype, copy=False)`) and use `np.einsum('ij,ijk->jk', casted_mask, array)` to entirely skip the intermediate 3D array allocation, significantly improving runtime.
## 2025-05-19 - Dot product scalar gradients allocation
**Learning:** During gradient calculation, `float((e * (-gamma * distance)).sum())` creates two full-size `(N, J)` arrays: one for the scaled distance and one for the element-wise multiplication before reduction.
**Action:** Replace `(A * B).sum()` with `np.vdot(A, B)` when scalar reduction is needed over matrix multiplication (where `B` can incorporate scalars naturally like `-gamma * np.vdot(A, B)`). This entirely avoids the 2D array allocation overhead and yields order-of-magnitude improvements in scalar gradient components.
## 2025-05-19 - Matrix Multiplication for Factor Gradient Calculation
**Learning:** Calculating gradients for factor arrays using boolean element-wise extraction and reduction like `(e * theta[:, factors]).sum(axis=0)` loops through columns and allocates large intermediate arrays (N x J). For large N (e.g., 5000), this significantly degrades performance.
**Action:** Replace looped subset extraction and element-wise products with highly optimized dense matrix multiplications combined with advanced integer indexing. For example, replace `(e * theta[:, factors]).sum(axis=0)` with `(e.T @ theta)[np.arange(e.shape[1]), factors]` to bypass intermediate (N x J) allocation, achieving massive execution speedups.
