## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** In highly mathematical Python operations like those in `fast_mlsirm/objective.py` and `math.py`, intermediate array allocation and boolean indexing (e.g. `x[x >= 0]`) act as significant performance bottlenecks because they copy memory. Similarly, using `np.sum(x ** 2, axis=1)` creates an intermediate squared array.
**Action:** Use advanced numpy vectorization to skip allocations. Replace boolean masking and element-wise assignment with functions like `np.clip` or `np.logaddexp` which are highly optimized in C. Replace reductions over intermediate arrays with `np.einsum` (e.g., `np.einsum('ij,ij->i', xi, xi)`) which skips the intermediate allocation.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** `np.sum(x * x)` 패턴은 파이썬 내에서 곱셈을 위한 새로운 중간 배열을 메모리에 할당하고 이후에 그 배열의 합을 구하게 되어 성능 저하를 야기합니다. `np.vdot(x, x)` (또는 `np.einsum`)를 활용하면 중간 메모리 할당을 우회할 수 있어 속도가 비약적으로 증가합니다.
**Action:** 거대한 배열의 크기나 요소 수와 관련된 최적화 시, `np.sum(x * x)` 대신 `np.vdot(x, x)`를 사용해 오버헤드를 방지합니다.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization in Basic Math
**Learning:** `sigmoid`, `softplus` 같은 기본 활성화 함수 내부에서 `1.0 / (1.0 + np.exp(-x))`나 `np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))`와 같은 연산은 매 과정마다 거대한 중간 배열(intermediate arrays) 복사를 야기하여 메모리 할당(allocation overhead)을 발생시킵니다.
**Action:** 이를 해결하기 위해 `out=` 파라미터를 활용한 in-place 연산(예: `np.exp(out, out=out)`)을 사용하면 메모리 정점(Peak Memory)을 획기적으로 줄이고 실행 속도 또한 상당히 개선할 수 있습니다.
