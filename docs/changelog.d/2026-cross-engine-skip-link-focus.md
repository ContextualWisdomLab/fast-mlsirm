# Cross-engine report skip-link focus repair

- Added a keyboard-visible skip link targeting `main#main-content[tabindex="-1"]`.
- Removed the target-specific `main:focus:not(:focus-visible)` outline suppressor so fragment/programmatic focus is not hidden by browser modality heuristics.
- Added a `prefers-reduced-motion: reduce` override that disables the skip-link transition.
- Added source contracts that reject focus suppression and require the reduced-motion boundary.

This is source-level evidence only. Chromium, Firefox, WebKit, assistive-technology, touch/keyboard, 320/768/desktop, eight-locale, and large-report median/p95 evidence remain required before merge.
