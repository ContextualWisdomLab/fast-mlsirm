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

## 2026-08-18 - [Prevent subprocess hang DoS]
**Vulnerability:** External `subprocess.run` calls without timeouts can hang indefinitely during GitHub CLI network or provider failures, stalling repository automation.
**Learning:** Command duration is a separate resource bound from JSON size/depth. A bounded parser cannot terminate a child process that never returns.
**Prevention:** Supply an explicit timeout for external repository-automation subprocesses and convert `subprocess.TimeoutExpired` into stable fail-closed evidence rather than hanging indefinitely.

## 2026-08-21 - [Bounded Capture Pipe and Process-Tree Cleanup]
**Vulnerability:** Repository automation can deadlock or retain descendants when
stdout/stderr pipes are inherited by a child process after the direct command
exits. Unbounded diagnostics can also exhaust memory or hide the original
fail-closed error when cleanup signalling fails.

**Learning:** A subprocess boundary needs independent byte limits, one absolute
deadline, concurrent pipe draining, strict machine-output decoding, and cleanup
that reaps the owned child without assuming signal delivery always succeeds.

**Prevention:** Keep stdout and stderr bounded, terminate the POSIX process
group when a reader proves a descendant owns a capture pipe, bounded-reap the
direct child, catch cleanup `OSError`, and preserve stable timeout/overflow/data
errors for governance and procurement evidence.

## 2026-09-12 - Fix insecure deserialization in _contract_object
**Vulnerability:** Untrusted JSON input deserialization in `_contract_object` lacked hooks to reject duplicate object keys and non-finite constant numbers (`NaN`, `Infinity`).
**Learning:** Python's `json.loads` is overly permissive by default. Without explicit `object_pairs_hook` and `parse_constant`, it can lead to JSON smuggling, logic bugs, or unintended decoder extensions, violating strict interoperable JSON expectations.
**Prevention:** When using `json.loads` to deserialize untrusted JSON in Python, always explicitly provide an `object_pairs_hook` to reject duplicate keys and a `parse_constant` hook to reject non-finite numbers.
## 2026-09-12 - Reject Numeric Overflow During JSON Deserialization
**Vulnerability:** Even when using `parse_constant` to reject `NaN` and `Infinity`, `json.loads` can still deserialize floating point numbers that evaluate to Infinity due to overflow (e.g., `1e999`).
**Learning:** `parse_constant` only intercepts explicit JSON literal constants like `NaN` or `Infinity`. Standard numeric values that exceed float limits silently become `inf` when parsed by default in Python.
**Prevention:** In addition to `parse_constant`, always provide a `parse_float` hook to `json.loads` that explicitly converts strings to floats and validates them using `math.isfinite()`.

## 2026-09-23 - JSON 깊이 검증 언더플로우 오탐 및 JSONDecodeError 방어 기제
**Vulnerability:** fast-mlsirm 내의 JSON 깊이 검증 함수들에서 닫는 괄호(`]}`)를 만날 때 `depth` 카운터가 0 미만으로 언더플로우되는 버그가 있었습니다. 초기에는 이를 통해 깊이 제한을 우회하여 `json.loads`에서 `RecursionError`를 유발할 수 있는 CRITICAL DoS 취약점으로 판단했습니다. 하지만 `]]]]]{"a":...}`와 같이 닫는 괄호가 앞에 오는 구조는 유효하지 않은 JSON이므로, `json.loads`가 파싱을 시작하자마자 깊은 탐색을 수행하기 전에 `JSONDecodeError`를 발생시켜 실행을 즉시 중단합니다. 따라서 실제 `RecursionError`나 자원 고갈로 이어지지 않으므로 심각한 취약점이 아닙니다.
**Learning:** 잘못된 형식의 JSON(예: 닫는 괄호로 시작하는 경우)은 `json.loads`가 구조를 깊게 탐색하기 전에 구문 오류(`JSONDecodeError`)로 거부합니다. 따라서 깊이 카운터 언더플로우가 존재하더라도 이것이 항상 DoS 취약점으로 이어지는 것은 아니며, 파서의 초기 검증 단계가 강력한 방어 기제로 작용할 수 있습니다.
**Prevention:** 취약점을 평가할 때는 전처리 로직의 버그가 실제 백엔드 엔진(`json.loads`)에서 어떻게 처리되는지 끝까지 검증해야 합니다. 카운터 언더플로우를 방지하기 위해 `and depth:` 조건을 추가하는 것은 올바른 조치이지만, 실제 익스플로잇 가능성을 과장하지 않도록 주의해야 합니다.
