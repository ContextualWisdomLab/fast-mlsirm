## 2024-07-07 - Add definition lists to Fit Report metric cards

**Learning:** Key-value pairs for metrics and statistics in HTML reports must use semantic `<dl>`, `<dt>`, and `<dd>` elements to properly associate the label with its value for screen readers. Using generic `<article>` or `<div>` with `<span>` and `<strong>` separates the semantic relationship. Additionally, default `<dt>` styling varies across browsers, so explicit `font-weight` resets are necessary to maintain cross-browser visual consistency while improving semantics. Also, relying purely on mouse interactions for focused tables (`:focus`) impacts mouse users visually - `:focus-visible` ensures focus outlines appear only during keyboard navigation. Purely visual charting elements (like horizontal bar tracks) inside HTML need `aria-hidden="true"` to prevent redundant, noisy announcements for screen reader users.
**Action:** Always prefer definition lists (`<dl>`) over generic markup for metric pairs, and explicitly reset their font-weight in CSS. Use `:focus-visible` instead of `:focus` for generic focus styles, and add `aria-hidden="true"` to decorative/visual chart tracks.

## 2025-02-18 - CLI Developer Experience
**Learning:** Even CLI tools benefit greatly from UX improvements (Developer Experience). Adding help strings and success feedback makes the tool much more intuitive.
**Action:** Always check if argparse/CLI tools have descriptive help texts and user feedback on success/failure.

## 2025-02-18 - CLI Error Handling DX
**Learning:** Raw stack traces from FileNotFoundError or exception in CLI tools provide poor developer experience.
**Action:** Wrap file loading and data processing steps in CLI applications with try-except blocks to output clean, user-friendly error messages to stderr.

## 2025-02-18 - CLI `simulate` Error Handling DX
**Learning:** Raw stack traces from `ValueError` (like invalid configuration parameters) during the `simulate` CLI command provide poor developer experience.
**Action:** Wrapped configuration validation and simulation execution steps in a try-except block to catch `ValueError` and `OSError`, outputting a clean, user-friendly error message to stderr and returning 1 to prevent raw stack traces.
