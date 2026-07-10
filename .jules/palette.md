## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.

## 2025-02-18 - CLI Error Handling DX
**Learning:** Raw stack traces from FileNotFoundError or exception in CLI tools provide poor developer experience.
**Action:** Wrap file loading and data processing steps in CLI applications with try-except blocks to output clean, user-friendly error messages to stderr.

## 2025-02-18 - CLI `simulate` Error Handling DX
**Learning:** Raw stack traces from `ValueError` (like invalid configuration parameters) during the `simulate` CLI command provide poor developer experience.
**Action:** Wrapped configuration validation and simulation execution steps in a try-except block to catch `ValueError` and `OSError`, outputting a clean, user-friendly error message to stderr and returning 1 to prevent raw stack traces.

## 2025-02-23 - HTML Report Structural and Interaction Polish
**Learning:** In dynamically generated HTML reports, standard `<article>`, `<span>`, and `<strong>` elements do not convey key-value semantics reliably to screen readers, whereas `<dl>`, `<dt>`, and `<dd>` provide explicit meaning. Visual-only tracks in complex components (like progress bars or value meters) without `aria-hidden="true"` often confuse assistive technologies. Finally, relying on `:focus` for container wrappers triggers an unwanted, sticky focus outline on mouse click interactions, whereas `:focus-visible` gracefully ensures focus rings appear exclusively during keyboard navigation.
**Action:** Always prefer semantic definition lists (`<dl>`) for arbitrary metric or key-value data displays. Explicitly use `:focus-visible` rather than `:focus` in CSS to ensure focus outlines appear only during keyboard navigation and not on mouse clicks, and apply `aria-hidden="true"` to inherently redundant visual tracks inside composite chart or metric rows.
