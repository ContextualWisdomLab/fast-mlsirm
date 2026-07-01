## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.

## 2025-02-18 - CLI Error Handling DX
**Learning:** Raw stack traces from FileNotFoundError or exception in CLI tools provide poor developer experience.
**Action:** Wrap file loading and data processing steps in CLI applications with try-except blocks to output clean, user-friendly error messages to stderr.

## 2025-02-12 - Graceful CLI Error Handling
**Learning:** For a tool focused on Developer Experience (DX), unhandled Python stack traces on user configuration errors (like negative numbers of persons or invalid models) can cause confusion and frustration. Gracefully catching expected exceptions (`ValueError`) and bubbling up friendly, descriptive errors significantly enhances the perceived polish and usability of the CLI.
**Action:** Always wrap top-level command implementations in try/except blocks to format parameter/configuration validation errors cleanly to stderr, preventing raw stack traces from reaching end users.
