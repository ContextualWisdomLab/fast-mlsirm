## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** In highly mathematical Python operations like those in `fast_mlsirm/objective.py` and `math.py`, intermediate array allocation and boolean indexing (e.g. `x[x >= 0]`) act as significant performance bottlenecks because they copy memory. Similarly, using `np.sum(x ** 2, axis=1)` creates an intermediate squared array.
**Action:** Use advanced numpy vectorization to skip allocations. Replace boolean masking and element-wise assignment with functions like `np.clip` or `np.logaddexp` which are highly optimized in C. Replace reductions over intermediate arrays with `np.einsum` (e.g., `np.einsum('ij,ij->i', xi, xi)`) which skips the intermediate allocation.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** `np.sum(x * x)` 패턴은 파이썬 내에서 곱셈을 위한 새로운 중간 배열을 메모리에 할당하고 이후에 그 배열의 합을 구하게 되어 성능 저하를 야기합니다. `np.vdot(x, x)` (또는 `np.einsum`)를 활용하면 중간 메모리 할당을 우회할 수 있어 속도가 비약적으로 증가합니다.
**Action:** 거대한 배열의 크기나 요소 수와 관련된 최적화 시, `np.sum(x * x)` 대신 `np.vdot(x, x)`를 사용해 오버헤드를 방지합니다.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization in Basic Math
**Learning:** `np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))`와 같은 수식은 중간에 많은 임시 배열을 생성하여 메모리 할당량(Peak memory)을 크게 증가시키며, Pandas와 같은 다른 데이터 객체를 다룰 때 가독성을 훼손하는 복잡한 in-place 연산으로 우회할 경우 타입 호환성이 깨질 위험이 높습니다.
**Action:** NumPy의 내장 C 구현체인 `np.logaddexp(0.0, x)`를 활용하면 중간 배열 생성 없이 최적화된 속도와 메모리 효율을 얻을 수 있으며 코드 가독성 또한 높일 수 있습니다.
