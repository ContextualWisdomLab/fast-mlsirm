# Supplemental exact-value report tooltips

## Added

- Added native `title` tooltips to finite floating-point metric cards, decorative
  bar labels, and diagnostic table cells so pointer users can inspect the
  unrounded Python float representation when the visible report uses compact
  significant-digit formatting.
- Preserved the existing accessible exact-value disclosure and JSON/CSV exports as
  the authoritative keyboard, touch, and assistive-technology paths; native title
  tooltips are supplemental and are not treated as an accessibility substitute.
- Added deterministic metric, chart, table, finite-value, non-finite-value, and
  non-float tests for the report tooltip contract.
