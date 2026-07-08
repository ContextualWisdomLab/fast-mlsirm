## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.

## 2025-02-18 - CLI Error Handling DX
**Learning:** Raw stack traces from FileNotFoundError or exception in CLI tools provide poor developer experience.
**Action:** Wrap file loading and data processing steps in CLI applications with try-except blocks to output clean, user-friendly error messages to stderr.

## 2025-02-18 - CLI `simulate` Error Handling DX
**Learning:** Raw stack traces from `ValueError` (like invalid configuration parameters) during the `simulate` CLI command provide poor developer experience.
**Action:** Wrapped configuration validation and simulation execution steps in a try-except block to catch `ValueError` and `OSError`, outputting a clean, user-friendly error message to stderr and returning 1 to prevent raw stack traces.

## 2025-02-18 - Table Container Focus States
**Learning:** Using `:focus` on generic focusable containers (like scrollable elements with `tabindex="0"`) creates a visually jarring outline when mouse users click on them.
**Action:** Always use `:focus-visible` instead of `:focus` for generic focusable containers to preserve required focus indicators for keyboard navigation without disrupting mouse users.
