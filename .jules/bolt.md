## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** In highly mathematical Python operations like those in `fast_mlsirm/objective.py` and `math.py`, intermediate array allocation and boolean indexing (e.g. `x[x >= 0]`) act as significant performance bottlenecks because they copy memory. Similarly, using `np.sum(x ** 2, axis=1)` creates an intermediate squared array.
**Action:** Use advanced numpy vectorization to skip allocations. Replace boolean masking and element-wise assignment with functions like `np.clip` or `np.logaddexp` which are highly optimized in C. Replace reductions over intermediate arrays with `np.einsum` (e.g., `np.einsum('ij,ij->i', xi, xi)`) which skips the intermediate allocation.

## 2024-05-19 - NumPy Array Allocation and Advanced Vectorization
**Learning:** `np.sum(x * x)` 패턴은 파이썬 내에서 곱셈을 위한 새로운 중간 배열을 메모리에 할당하고 이후에 그 배열의 합을 구하게 되어 성능 저하를 야기합니다. `np.vdot(x, x)` (또는 `np.einsum`)를 활용하면 중간 메모리 할당을 우회할 수 있어 속도가 비약적으로 증가합니다.
**Action:** 거대한 배열의 크기나 요소 수와 관련된 최적화 시, `np.sum(x * x)` 대신 `np.vdot(x, x)`를 사용해 오버헤드를 방지합니다.
## 2025-03-09 - boolean 배열 `.sum()` 검사의 성능 최적화
**Learning:** `np.any(observed.sum(axis=0) == 0)` 와 같이 boolean 배열에 대해 `.sum()`을 호출하는 경우 boolean 값이 정수(integer)로 캐스팅되는 오버헤드가 발생한다. 이로 인해 불필요한 성능 저하가 일어난다. `benchmark.py` 에서 확인한 바, `sum` 방식은 100회 실행 시 약 2.2초 소요되나 `any` 방식은 0.2초 소요되었다(약 10배 차이).
**Action:** boolean 배열의 축(axis)을 기준으로 값이 존재하는지 검사할 때는 `.sum()`을 사용하지 말고, `.any()` 와 `.all()` 을 조합하여 `not np.all(observed.any(axis=0))` 형태로 작성해야 한다.
