# Essay HTML Report Print, Focus, and Tooltip Accessibility

## Scope

This record governs the micro-UX modifications applied to the automated-essay HTML scoring, validation, and calibration reports. The artifact remains serialized, script-free HTML. These changes do not modify any psychometric calculation, modeling logic, consequential-decision policy, or the underlying dataset JSON properties.

## Standards decision

1. **Unrounded Numeric Tooltips:** For metrics and table cells presenting truncated float values (e.g., restricted to 4 significant digits), we expose the precise full-precision `repr()` evaluation via the native `title` HTML attribute. This provides mouse/pointer users with an easy way to read unrounded exact mathematical output without JS. However, `title` alone is notoriously inaccessible for keyboard, screen-reader, or touch users, and therefore we make *no WCAG-conformance claim* that this tooltip resolves WCAG 1.3.1 (Info and Relationships) or 4.1.2 (Name, Role, Value). Instead, the full-precision `exact_values` table and the deterministic JSON export remain the authoritative accessibility boundary for this data.
2. **Fallback Focus Visibility:** We retain the explicit `main:focus-visible { outline: 3px solid Highlight; }` while leaving standard focus behavior intact for backward compatibility. We intentionally avoid `main:focus { outline: none; }` which would destroy fallback focus rings for browsers without `:focus-visible` support.
3. **Print Optimization:** Browsers automatically convert colors and disable backgrounds during print or PDF export. We added `@media print` rules setting text to black-on-white, avoiding awkward page/section breaks with `break-inside: avoid`, and hiding interactive elements (like the skip-to-content link) that are useless on paper using `display: none !important;`.

## Verification contract

Exact regressions in the test suite enforce that `title` attributes are applied strictly and accurately to finite `float` instances (while skipping `None`, integers, or strings). The exact HTML assertions enforce that the `@media print` rules are present and that the unsafe `main:focus { outline: none; }` rule is omitted from the stylesheet.
