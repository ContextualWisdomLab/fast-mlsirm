# Explicit criterion row headers in governed HTML reports

## Changed

- Replaced positional first-column row-header inference with an explicit
  zero-based `row_header_column` contract in the governed essay score renderer.
- Emit `<th scope="row">` only for criterion identifiers; evidence-reference
  values remain data cells, and malformed header/row widths fail closed.
- Added complete-artifact parsing tests that verify table semantics and exact
  canonical JSON reconstruction, plus APA 7th WCAG 2.2 doctoring.
