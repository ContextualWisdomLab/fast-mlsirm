# Report bar-row motion and tab-order repair

- Keeps the background-color hover transition and disables it under `prefers-reduced-motion: reduce`.
- Keeps exact labels and values in structured report content.
- Rejects synthetic `tabindex="0"`, `role="region"`, and target-specific focus suppression on informational bar rows so large reports do not add one tab stop and landmark per data row.
- Adds source contracts for reduced motion and bounded keyboard semantics.

Real Chromium, Firefox, WebKit, assistive-technology, touch/keyboard, 320/768/desktop, eight-locale, and large-report median/p95 evidence remain required before merge.
