## 2025-02-28 - [CLI Stack Trace Leakage Prevention]
**Vulnerability:** The CLI entrypoint (`python/fast_mlsirm/cli.py`) did not wrap its core execution in a high-level exception handler. Any error (e.g., missing file, invalid input format) would crash the program and leak the raw Python stack trace and internal file paths to the console.
**Learning:** Even in local CLI tools, exposing raw stack traces breaks the "fail securely" principle and degrades the user experience by leaking internal implementations.
**Prevention:** Always wrap the highest-level execution logic (like a `main()` function or CLI dispatcher) in a generic `try...except` block to catch unhandled exceptions, print a sanitized and clear error message to `stderr`, and exit with a non-zero code.
