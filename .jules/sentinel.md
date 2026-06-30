## 2026-06-29 - [np.load Insecure Deserialization Risk & Assertion Optimization Removal]
**Vulnerability:**
1. `numpy.load()` was used without explicitly specifying `allow_pickle=False`. This could lead to insecure deserialization and arbitrary code execution if a malicious pickle file is loaded (especially critical depending on the environment's NumPy version).
2. `assert` was used for critical control flow (`assert best is not None`). Assertions are stripped out when Python is run with the `-O` optimization flag, potentially leading to undefined behavior and masking errors in production environments.

**Learning:**
Explicitly defining `allow_pickle=False` is a robust defense-in-depth practice. Relying on `assert` for necessary runtime checks is dangerous; standard exceptions like `RuntimeError` should be used instead.

**Prevention:**
- Always add `allow_pickle=False` to `np.load` unless explicitly required and verified.
- Replace critical `assert` statements with `if` condition checks that raise appropriate runtime exceptions.

## 2026-06-30 - [Missing error handling exposing stack traces]
**Vulnerability:** The CLI `main` execution logic was missing a global exception handler. In case of user errors (e.g. invalid arguments or missing files), internal application stack traces were printed directly to `sys.stderr`, leaking internal system states and paths.
**Learning:** Stack traces should never be exposed in production code interfaces unless explicitly enabled (e.g., via `--debug`). Unhandled exceptions in CLI commands lead to poor user experience and leak information.
**Prevention:**
- Always wrap main entry points for CLI applications in a `try...except` block.
- Print generalized, user-friendly error messages (e.g., `Error: [Errno 2] No such file or directory`) instead of leaking internals to stderr.
