## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.

## 2025-02-18 - CLI Error Handling DX
**Learning:** Raw stack traces from FileNotFoundError or exception in CLI tools provide poor developer experience.
**Action:** Wrap file loading and data processing steps in CLI applications with try-except blocks to output clean, user-friendly error messages to stderr.

## 2024-05-19 - Improved CLI Error Feedback for Simulate
**Learning:** Raw stack traces from `ValueError` (like invalid configuration constraints) during CLI subcommand execution provide poor UX. Users need clear, actionable error messages instead of raw Python exceptions.
**Action:** When implementing CLI subcommands in `fast_mlsirm/cli.py`, always wrap the core execution logic in try-except blocks. Catch `ValueError` to provide user-friendly configuration validation messages (e.g., printed to sys.stderr) and return a non-zero exit code.
