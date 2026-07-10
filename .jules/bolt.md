## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** In highly mathematical Python operations like those in `fast_mlsirm/objective.py` and `math.py`, intermediate array allocation and boolean indexing (e.g. `x[x >= 0]`) act as significant performance bottlenecks because they copy memory. Similarly, using `np.sum(x ** 2, axis=1)` creates an intermediate squared array.
**Action:** Use advanced numpy vectorization to skip allocations. Replace boolean masking and element-wise assignment with functions like `np.clip` or `np.logaddexp` which are highly optimized in C. Replace reductions over intermediate arrays with `np.einsum` (e.g., `np.einsum('ij,ij->i', xi, xi)`) which skips the intermediate allocation.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** `np.sum(x * x)` 패턴은 파이썬 내에서 곱셈을 위한 새로운 중간 배열을 메모리에 할당하고 이후에 그 배열의 합을 구하게 되어 성능 저하를 야기합니다. `np.vdot(x, x)` (또는 `np.einsum`)를 활용하면 중간 메모리 할당을 우회할 수 있어 속도가 비약적으로 증가합니다.
**Action:** 거대한 배열의 크기나 요소 수와 관련된 최적화 시, `np.sum(x * x)` 대신 `np.vdot(x, x)`를 사용해 오버헤드를 방지합니다.

## 2024-03-10 - Vectorize looping over dimension

**Learning:** When calculating values grouped by categorical dimensions (like item factors), calculating sums or means by iterating through dimensions `d in range(n_dims)` using boolean indexing `(items = factor_id == d)` is slow because numpy does not vectorize over the outer loop. Utilizing a 2D boolean mapping mask `(factor_id[:, None] == np.arange(n_dims))` and matrix multiplications `(@)` directly converts loop aggregations into fast C/BLAS optimized operations, yielding massive performance gains.

**Action:** Whenever noticing an outer python loop over categorical subsets (often indices or distinct labels) that aggregates numeric array data, immediately attempt to broadcast the labels into a dense or sparse 2D boolean mask and aggregate using matrix multiplication `(@)`.

## 2024-07-10 - Optimizing Distance Computation
**Learning:** When calculating pairwise distances between `xi` (shape N x D) and `zeta` (shape J x D), computing `((xi[:, None, :] - zeta[None, :, :]) ** 2).sum(axis=2)` involves 3D broadcasting taking O(N*J*D) memory allocation which is very slow. Expanding `(x-y)^2` into `x^2 + y^2 - 2xy` and computing as `xi_sq[:, None] + zeta_sq[None, :] - 2 * np.dot(xi, zeta.T)` reduces intermediate allocation to O(N*J) 2D array allocation and uses BLAS optimizations for matrix multiplication, leading to over 5x performance improvement.
**Action:** Identify and replace 3D pairwise distance broadcast computation with 2D array allocation dot-products.
