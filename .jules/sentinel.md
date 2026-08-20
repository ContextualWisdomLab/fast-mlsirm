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
## 2024-08-20 - [Subprocess JSON DoS Vulnerability via Unbounded Deserialization]
**Vulnerability:** Scripts (`scripts/build_pr_queue_governance.py` and `scripts/build_procurement_due_diligence.py`) directly used `json.loads(completed.stdout)` to deserialize output from subprocesses. This unbounded parsing could lead to a Denial of Service (DoS) attack if a compromised or misconfigured external service returned excessively large or deeply nested JSON, exhausting memory or recursion limits.
**Learning:** Data arriving from subprocess stdout must be treated as untrusted input. Just like reading from external files, unbounded JSON decoding from subprocess stdout can trigger `MemoryError` or `RecursionError` and crash the automation pipeline.
**Prevention:** Always use a bounded JSON parser like `parse_json_bounded` that explicitly checks length and structural depth before invoking the standard `json.loads` decoder, even for data originating from subprocess standard output.
