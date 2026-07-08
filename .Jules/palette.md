## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.

## 2025-02-18 - CLI Error Handling DX
**Learning:** Raw stack traces from FileNotFoundError or exception in CLI tools provide poor developer experience.
**Action:** Wrap file loading and data processing steps in CLI applications with try-except blocks to output clean, user-friendly error messages to stderr.

## 2025-02-18 - CLI `simulate` Error Handling DX
**Learning:** Raw stack traces from `ValueError` (like invalid configuration parameters) during the `simulate` CLI command provide poor developer experience.
**Action:** Wrapped configuration validation and simulation execution steps in a try-except block to catch `ValueError` and `OSError`, outputting a clean, user-friendly error message to stderr and returning 1 to prevent raw stack traces.

## 2025-02-18 - CLI Version Flag DX
**Learning:** Adding a `--version` flag to a CLI application is a critical DX improvement that allows users and developers to quickly verify the installed package version without needing to inspect Python environments.
**Action:** Always add `--version` arguments to `argparse` setups using `action="version"` and pulling the version from the package's `__init__.py`.
