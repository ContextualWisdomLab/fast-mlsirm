## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.
## 2025-06-30 - Graceful CLI file validation
**Learning:** Raw Python tracebacks from `FileNotFoundError` during CLI execution degrade Developer Experience (DX) because they look like internal application crashes rather than user input errors.
**Action:** Validate file existence proactively using `pathlib.Path.exists()` before loading data, and return a clean, user-friendly error message to `sys.stderr` when required inputs are missing.
