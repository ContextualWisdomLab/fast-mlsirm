# Keep essay-report pointer focus modality-safe

## Fixed

- Suppress pointer-acquired outlines on focusable essay-report table regions and canonical JSON blocks only when `:focus-visible` is false. Keyboard navigation retains the explicit high-contrast focus indicator, and regressions reject blanket `:focus { outline: none; }` suppression.
