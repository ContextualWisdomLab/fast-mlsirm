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

## 2026-08-25 - [Strict JSON object and numeric-constant admission]
**Vulnerability:** `_contract_object` accepted duplicate object members and Python's non-standard `NaN` / `Infinity` / `-Infinity` constants. `_response_object` already rejected duplicate members on the protected baseline but still admitted those non-finite constants. The two trust boundaries therefore exposed different strict-JSON guarantees.
**Learning:** Python's `json.loads` accepts non-standard non-finite constants by default and resolves duplicate object members with last-key-wins semantics unless an `object_pairs_hook` rejects them. Security evidence must distinguish a pre-existing guard from the newly repaired behavior instead of attributing the same defect to both boundaries.
**Prevention:** Require `parse_constant` rejection at both boundaries and duplicate-member rejection wherever the protected baseline does not already provide it. Keep focused regression coverage for `NaN`, `Infinity`, `-Infinity`, and duplicate members, and avoid unsupported DoS/cache-poisoning claims unless a concrete execution path demonstrates them.
