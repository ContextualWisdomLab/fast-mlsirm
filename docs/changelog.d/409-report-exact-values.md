# Accessible exact-value disclosure in HTML reports

## Added

- A shared `report_exact_values` component that renders every plotted report
  section's complete, untruncated source rows as an open-by-default native
  `details`/`summary` disclosure with a semantic full-precision table
  (float cells use shortest round-trip `repr`, missing cells are named
  explicitly) plus copyable JSON and CSV exports generated from the same
  rows (strict JSON with `null` missingness; RFC 4180 CSV).
- Fit and dimensionality report sections now carry the disclosure adjacent
  to their charts, so exact values stay available on touch input, in
  print/PDF output, under keyboard-only navigation, and with JavaScript
  disabled, per WCAG 2.2 success criteria 1.3.1, 1.4.13, 2.1.1, and 4.1.2
  (issue #409). Chart markup and the summarized 12-row tables are unchanged.
