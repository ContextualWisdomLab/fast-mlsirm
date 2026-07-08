## 2026-06-29 - [np.load Insecure Deserialization Risk & Assertion Optimization Removal]
**Vulnerability:**
1. `numpy.load()` was used without explicitly specifying `allow_pickle=False`. This could lead to insecure deserialization and arbitrary code execution if a malicious pickle file is loaded (especially critical depending on the environment's NumPy version).
2. `assert` was used for critical control flow (`assert best is not None`). Assertions are stripped out when Python is run with the `-O` optimization flag, potentially leading to undefined behavior and masking errors in production environments.

**Learning:**
Explicitly defining `allow_pickle=False` is a robust defense-in-depth practice. Relying on `assert` for necessary runtime checks is dangerous; standard exceptions like `RuntimeError` should be used instead.

**Prevention:**
- Always add `allow_pickle=False` to `np.load` unless explicitly required and verified.
- Replace critical `assert` statements with `if` condition checks that raise appropriate runtime exceptions.

## 2026-07-02 - [Add Strict CSP to HTML Reports]
**Vulnerability:**
The generated standalone HTML diagnostic reports lacked a Content Security Policy (CSP). While the backend handles escaping HTML input, missing a CSP violates defense-in-depth principles. If any unsanitized user data were processed or if future dynamic features were introduced without escaping, the reports could be vulnerable to Cross-Site Scripting (XSS).

**Learning:**
Even for static, offline-generated HTML reports where inputs are meant to be controlled, implementing a strict CSP is a crucial layer of security.

**Prevention:**
- Always add a strict CSP `<meta>` tag (e.g., `default-src 'none'; style-src 'unsafe-inline';`) to the `<head>` of generated standalone HTML files.
