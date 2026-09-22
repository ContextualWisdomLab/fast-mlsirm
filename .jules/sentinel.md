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
## 2026-09-12 - Reject Numeric Overflow During JSON Deserialization
**Vulnerability:** Even when using `parse_constant` to reject `NaN` and `Infinity`, `json.loads` can still deserialize floating point numbers that evaluate to Infinity due to overflow (e.g., `1e999`). This bypasses literal checks and can introduce invalid float states.
**Learning:** `parse_constant` only intercepts explicit JSON literal constants like `NaN` or `Infinity`. Standard numeric values that exceed Python's float limits silently become `inf` when parsed by default.
**Prevention:** In addition to `parse_constant`, always provide a `parse_float` hook to `json.loads` that explicitly converts string representations to floats and validates them using `math.isfinite()`. Ensure the necessary modules like `math` are imported at the top level to avoid overhead in the parsing hot-path.
## 2026-09-20 - Fix Semgrep SAST vulnerabilities (dangerous-globals-use, non-literal-import)
**Vulnerability:** The Semgrep CI check identified two Medium+ SAST vulnerabilities:
1. `dangerous-globals-use` in `fast_mlsirm/dif.py`: Dynamic dictionary lookups via `globals()` with string keys are flagged as a potential code execution vector.
2. `non-literal-import` in `tools/inventory_public_api.py`: Dynamic module imports via `importlib.import_module()` based on untrusted/unconstrained string input can lead to arbitrary code execution.
**Learning:** Security analysis tools like Semgrep strictly enforce best practices. In `dif.py`, `globals()` isn't necessary when we can reference the functions directly in a loop. In internal scripts, dynamic imports are sometimes necessary, but they must be properly sandboxed/whitelisted to prove they only act on expected internal code.
**Prevention:**
1. Avoid `globals()` where possible. If mapping strings to functions is needed, construct an explicit `dict` containing the allowed function references.
2. For dynamic imports, implement an explicit prefix/whitelist check (e.g., `modname.startswith("fast_mlsirm.")`) before calling `importlib.import_module()` to assure the static analyzer the input is bounded to safe paths.
## 2026-09-21 - Suppress non-literal-import SAST Warning
**Vulnerability:** Semgrep flags `importlib.import_module(modname)` with `non-literal-import` when `modname` is dynamically generated, warning of arbitrary code execution.
**Learning:** While explicit whitelist prefixing (e.g., `if not modname.startswith("fast_mlsirm.")`) makes the dynamic import safe at runtime, Semgrep's static analysis engine is not always sophisticated enough to infer that this guard is sufficient to clear the warning.
**Prevention:** In internal scripts where dynamic imports are intentionally used and correctly guarded by whitelists, use an explicit inline `# nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import` comment to suppress the false positive warning and pass CI.
## 2026-09-22 - Optimize json.loads parse_float performance
**Vulnerability:** `json.loads`의 `parse_float` 콜백이나, 파라미터 매핑을 위해 내부 함수 내부에 `import math`가 위치하면 각각의 부동 소수점을 파싱/처리할 때마다 모듈 로드 오버헤드가 발생합니다. 반복문/콜백 내부의 `import`는 Python 모듈 캐시(`sys.modules`)를 조회하지만 핫-패스에서는 무시할 수 없는 상당한 지연이 발생할 수 있습니다.
**Learning:** 엄청난 양의 부동 소수점을 갖는 입력이나 대규모 데이터 세트 변환 중, 불필요한 import overhead는 전체 처리 시간에 상당한 병목 현상을 일으켜 성능을 저하시키고 DoS 위험성을 내포합니다.
**Prevention:** `import math`와 같은 모듈 임포트 구문은 내부 함수가 아닌 모듈의 최상단(top-level)에 위치시켜 매번 모듈을 로드하는 오버헤드를 방지해야 합니다.
