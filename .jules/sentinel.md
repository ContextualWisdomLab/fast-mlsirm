## 2026-06-29 - [np.load Insecure Deserialization Risk & Assertion Optimization Removal]
**Vulnerability:**
1. `numpy.load()` was used without explicitly specifying `allow_pickle=False`. This could lead to insecure deserialization and arbitrary code execution if a malicious pickle file is loaded (especially critical depending on the environment's NumPy version).
2. `assert` was used for critical control flow (`assert best is not None`). Assertions are stripped out when Python is run with the `-O` optimization flag, potentially leading to undefined behavior and masking errors in production environments.

**Learning:**
Explicitly defining `allow_pickle=False` is a robust defense-in-depth practice. Relying on `assert` for necessary runtime checks is dangerous; standard exceptions like `RuntimeError` should be used instead.

**Prevention:**
- Always add `allow_pickle=False` to `np.load` unless explicitly required and verified.
- Replace critical `assert` statements with `if` condition checks that raise appropriate runtime exceptions.

## 2026-07-06 - [DoS via Unconstrained Array Dimension Allocation]
**Vulnerability:** In `fast_mlsirm/fit.py`, the number of dimensions `n_dims` was calculated using the maximum value provided in user input (`factor_id.max()`). A maliciously crafted large integer in `factor_id` causes `np.zeros((n_persons, n_dims))` to attempt allocating an impossibly large array (e.g. hundreds of GiB), crashing the application via Out-Of-Memory (OOM) and causing a Denial of Service (DoS).
**Learning:** Never trust user input to define unconstrained array dimensions, especially when derived from maximum values within the data.
**Prevention:** Add explicit boundary checks (e.g. `n_dims > n_items`) to ensure derived dimensions remain mathematically sound and computationally feasible before memory allocation.
## 2024-07-04 - [Defense in Depth] Validate URI Schemes in Link Generation
**Vulnerability:** A script (`scripts/build_pr_queue_governance.py`) used `escape()` to sanitize URLs placed directly in the `href` attribute of an `<a>` tag. However, `escape()` alone is insufficient to prevent XSS if the URL uses an unsafe protocol such as `javascript:` or `data:`.
**Learning:** This is a classic case where escaping HTML special characters provides a false sense of security for URI-based injection contexts. An attacker could potentially inject a malicious script by providing an unsafe protocol.
**Prevention:** Always validate URI schemes and restrict them to safe protocols (e.g., `http:`, `https:`) before using them in contexts like `href` or `src`. If an unsafe scheme is detected, the URL should be neutralized (e.g., replaced with `#`). I implemented a `_safe_url` helper function to enforce this.
## 2026-07-12 - [Bandit B324: Use of weak MD5 hash for security]
**Vulnerability:** MD5 hashing in `fast_mlsirm/report.py` triggered a high severity warning by Bandit, because by default it is assumed to be used for security purposes which is unsafe due to weak hashing.
**Learning:** For non-security purposes like generating unique dom ids, `hashlib.md5()` triggers a vulnerability warning unless `usedforsecurity=False` is passed. This allows bypassing FIPS compliance limitations as well as suppressing false positive warnings.
**Prevention:** Always add `usedforsecurity=False` parameter to `hashlib.md5` and other weak hashing functions unless they are genuinely used for secure cryptography (which they shouldn't be).
## 2026-07-30 - [JSON Denial of Service (DoS) Vulnerability]
**Vulnerability:** The HTML report generator `fast_mlsirm/report.py` used `json.loads(source.read_text())` directly on potentially unconstrained diagnostics output. This presents a DoS risk where a malicious or malformed input JSON could trigger unbounded recursion (excessive nesting) or memory exhaustion (loading massive payloads into memory).
**Learning:** Directly using `json.loads()` on file contents bypasses size and depth limitations, making the application vulnerable to DoS attacks. The `_load_json_bounded` utility in `fast_mlsirm.io` provides a robust, defense-in-depth alternative by enforcing explicit size limits and depth checks before delegating to `json.loads()`.
**Prevention:** Never use `json.loads()` on unvalidated file input. Always utilize `_load_json_bounded` or a similar bounded deserialization utility to protect against memory exhaustion and unbounded recursion attacks.
## 2026-08-11 - [JSON Recursion DoS Vulnerability on String Deserialization]
**Vulnerability:** The functions `parse_generated_item_candidate` and `_contract_object` used `json.loads` directly on string payloads before strictly enforcing depth limits over the string itself. A maliciously nested JSON string (e.g. `{"a": {"a": ...}}`) could exceed the Python maximum recursion limit, crashing the process with a `RecursionError` and causing a Denial of Service (DoS) attack, because Python's built-in `json.loads` recurses natively while decoding.
**Learning:** Checking for JSON nested depth after decoding using `json.loads` (or implicitly relying on string size constraints) is insufficient to prevent recursion crashes on deep but compact objects. Depth checking must happen by scanning the raw string stream prior to any decoding engine invocations.
**Prevention:** Always implement a character-level depth limit scanner (`_validate_raw_json_depth`) and enforce it on raw strings before passing them to `json.loads`.
## 2026-08-15 - [하위 프로세스 출력에서 제한 없는 JSON 로드로 인한 메모리 소모 (DoS) 취약점]
**Vulnerability:** 여러 스크립트(`scripts/build_pr_queue_governance.py`, `scripts/audit_workflow_registry.py` 등)에서 하위 프로세스(subprocess)의 표준 출력(`stdout`) 결과를 파싱할 때 `json.loads`를 직접 사용했습니다. 악의적으로 조작되거나 예상치 못한 거대한 크기 및 깊은 중첩(nesting)을 가진 출력 결과가 전달될 경우, 메모리 고갈이나 파이썬의 최대 재귀(recursion) 한도 초과를 유발하여 애플리케이션의 크래시(DoS)가 발생할 수 있습니다.
**Learning:** 파일 내용뿐만 아니라 `subprocess.run` 등의 명령줄 도구 실행 결과나 외부 시스템과 상호 작용하며 전달받은 문자열 데이터를 파싱할 때도 항상 크기와 깊이의 한계가 명시적으로 설정되어야 합니다. 그렇지 않으면 메모리 고갈 공격에 취약해집니다.
**Prevention:** 텍스트나 문자열을 JSON으로 역직렬화하기 전에는 반드시 `_validate_raw_json_depth` 등을 포함하는 `parse_json_bounded` (또는 이와 유사한 안전한 함수)를 사용하여 데이터 크기와 중첩 깊이를 검증해야 합니다. 하위 프로세스의 반환값 역시 신뢰할 수 없는 데이터로 취급하십시오.
