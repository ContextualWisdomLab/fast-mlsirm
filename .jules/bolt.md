## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** In highly mathematical Python operations like those in `fast_mlsirm/objective.py` and `math.py`, intermediate array allocation and boolean indexing (e.g. `x[x >= 0]`) act as significant performance bottlenecks because they copy memory. Similarly, using `np.sum(x ** 2, axis=1)` creates an intermediate squared array.
**Action:** Use advanced numpy vectorization to skip allocations. Replace boolean masking and element-wise assignment with functions like `np.clip` or `np.logaddexp` which are highly optimized in C. Replace reductions over intermediate arrays with `np.einsum` (e.g., `np.einsum('ij,ij->i', xi, xi)`) which skips the intermediate allocation.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** `np.sum(x * x)` 패턴은 파이썬 내에서 곱셈을 위한 새로운 중간 배열을 메모리에 할당하고 이후에 그 배열의 합을 구하게 되어 성능 저하를 야기합니다. `np.vdot(x, x)` (또는 `np.einsum`)를 활용하면 중간 메모리 할당을 우회할 수 있어 속도가 비약적으로 증가합니다.
**Action:** 거대한 배열의 크기나 요소 수와 관련된 최적화 시, `np.sum(x * x)` 대신 `np.vdot(x, x)`를 사용해 오버헤드를 방지합니다.

## 2024-07-02 - NumPy `log1p` and `exp` intermediate array allocations
**Learning:** In heavily mathematical Python functions like `softplus`, executing mathematical formulas exactly as written (e.g., `np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))`) causes NumPy to allocate temporary arrays in memory for every intermediate operation (`np.abs`, `np.exp`, `np.log1p`, `np.maximum`). This memory overhead introduces a significant performance bottleneck.
**Action:** Replace composite math expressions with mathematically equivalent native NumPy functions whenever possible. For example, replace manual `softplus` formulas with `np.logaddexp(x.dtype.type(0.0), x)`, which operates on the C backend directly and avoids intermediate temporaries (besides the output array), resulting in >5x speed improvements.
