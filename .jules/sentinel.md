## 2026-06-29 - [np.load Insecure Deserialization Risk & Assertion Optimization Removal]
**Vulnerability:**
1. `numpy.load()` was used without explicitly specifying `allow_pickle=False`. This could lead to insecure deserialization and arbitrary code execution if a malicious pickle file is loaded (especially critical depending on the environment's NumPy version).
2. `assert` was used for critical control flow (`assert best is not None`). Assertions are stripped out when Python is run with the `-O` optimization flag, potentially leading to undefined behavior and masking errors in production environments.

**Learning:**
Explicitly defining `allow_pickle=False` is a robust defense-in-depth practice. Relying on `assert` for necessary runtime checks is dangerous; standard exceptions like `RuntimeError` should be used instead.

**Prevention:**
- Always add `allow_pickle=False` to `np.load` unless explicitly required and verified.
- Replace critical `assert` statements with `if` condition checks that raise appropriate runtime exceptions.

## 2025-02-21 - Add Content-Security-Policy to HTML Report
**Vulnerability:** Standalone HTML reports created by `render_diagnostics_report` lacked a Content-Security-Policy (CSP) meta tag, making them potentially vulnerable to XSS if the generated HTML file was hosted or opened in a context where malicious input could be injected into the payload.
**Learning:** Even generated offline or standalone artifacts should enforce a strict CSP (`default-src 'none'; style-src 'unsafe-inline';`) because users might host these files or share them in environments where XSS is exploitable.
**Prevention:** Include strict CSP meta tags in all dynamically generated HTML documents, especially those that include user-supplied or external data.
