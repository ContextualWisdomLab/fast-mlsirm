## 2026-06-29 - [np.load Insecure Deserialization Risk & Assertion Optimization Removal]
**Vulnerability:**
1. `numpy.load()` was used without explicitly specifying `allow_pickle=False`. This could lead to insecure deserialization and arbitrary code execution if a malicious pickle file is loaded (especially critical depending on the environment's NumPy version).
2. `assert` was used for critical control flow (`assert best is not None`). Assertions are stripped out when Python is run with the `-O` optimization flag, potentially leading to undefined behavior and masking errors in production environments.

**Learning:**
Explicitly defining `allow_pickle=False` is a robust defense-in-depth practice. Relying on `assert` for necessary runtime checks is dangerous; standard exceptions like `RuntimeError` should be used instead.

**Prevention:**
- Always add `allow_pickle=False` to `np.load` unless explicitly required and verified.
- Replace critical `assert` statements with `if` condition checks that raise appropriate runtime exceptions.
## 2026-07-01 - Add Content-Security-Policy to HTML reports
**Vulnerability:** Missing Content-Security-Policy (CSP) in dynamically generated standalone HTML reports.
**Learning:** `python/fast_mlsirm/report.py` generates local, standalone HTML reports. Although the inputs are numerical and JSON, adding a strict CSP (`default-src 'none'; style-src 'unsafe-inline'`) prevents unexpected resource loading and potential XSS if untrusted input is somehow passed in the future.
**Prevention:** Include restrictive CSP tags in all generated HTML, even for offline data visualization.
