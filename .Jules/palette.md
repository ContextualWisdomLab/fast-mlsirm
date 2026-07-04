## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.

## 2025-02-18 - CLI Error Handling DX
**Learning:** Raw stack traces from FileNotFoundError or exception in CLI tools provide poor developer experience.
**Action:** Wrap file loading and data processing steps in CLI applications with try-except blocks to output clean, user-friendly error messages to stderr.

## 2025-02-18 - CLI User Input Validation
**Learning:** Returning a raw "Unexpected failure" with a stack trace when a user simply enters an invalid configuration value (like --dims -1) provides a poor Developer Experience (DX).
**Action:** Always wrap business logic that can raise expected validation errors (like ValueError) in try/except blocks to print a friendly, actionable error message to stderr.
