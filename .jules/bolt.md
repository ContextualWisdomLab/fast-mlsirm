## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** In highly mathematical Python operations like those in `fast_mlsirm/objective.py` and `math.py`, intermediate array allocation and boolean indexing (e.g. `x[x >= 0]`) act as significant performance bottlenecks because they copy memory. Similarly, using `np.sum(x ** 2, axis=1)` creates an intermediate squared array.
**Action:** Use advanced numpy vectorization to skip allocations. Replace boolean masking and element-wise assignment with functions like `np.clip` or `np.logaddexp` which are highly optimized in C. Replace reductions over intermediate arrays with `np.einsum` (e.g., `np.einsum('ij,ij->i', xi, xi)`) which skips the intermediate allocation.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** `np.sum(x * x)` 패턴은 파이썬 내에서 곱셈을 위한 새로운 중간 배열을 메모리에 할당하고 이후에 그 배열의 합을 구하게 되어 성능 저하를 야기합니다. `np.vdot(x, x)` (또는 `np.einsum`)를 활용하면 중간 메모리 할당을 우회할 수 있어 속도가 비약적으로 증가합니다.
**Action:** 거대한 배열의 크기나 요소 수와 관련된 최적화 시, `np.sum(x * x)` 대신 `np.vdot(x, x)`를 사용해 오버헤드를 방지합니다.
## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization in math.py
**Learning:** `np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))`와 같은 기존 구현은 파이썬 레벨에서 여러 개의 중간 배열(`np.maximum`, `np.abs`, `np.exp`, `np.log1p`)을 임시로 메모리에 할당하여 병목을 유발합니다.
**Action:** `np.logaddexp(0.0, x)`를 사용하여 내부 C 구현 수준에서 중간 메모리 할당 없이 $\log(1 + e^x)$를 한 번에 계산하도록 처리했습니다.
