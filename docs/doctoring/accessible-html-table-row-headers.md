# Explicit row-header semantics for governed HTML reports

## Decision

The governed essay score renderer identifies a table-body header only when the
caller supplies an explicit zero-based `row_header_column`. The helper does not
infer that the first column is a row header from position alone.

The criterion-outcomes table supplies `row_header_column=0` because each
`criterion_id` identifies the observation row. The source-text-free evidence
table does not declare a row-header column: its first value is repeated
criterion provenance inside a many-row evidence relation rather than a unique
row identity.

The governed essay facets-calibration renderer likewise supplies
`row_header_column=0` for task, rater, respondent, ordered-threshold, and
likelihood-trace tables because those first-column values identify the row's
semantic axis. The validation-evidence renderer supplies it for the metric
identity column. This is an explicit domain decision at each call site rather
than a blanket first-column transform.

Every non-empty row must have exactly the same width as the declared header
axis. Invalid, Boolean, negative, or out-of-range row-header indices fail before
HTML is emitted. This prevents a malformed row from silently shifting a
`scope="row"` association onto the wrong exact value.

## Accessibility contract

The renderer emits:

- `<th scope="col">` for every declared column header;
- `<th scope="row">` only for an explicitly identified row-identity field;
- `<td>` for all other body values;
- a semantic `<caption>` for each table;
- a keyboard-focusable overflow region around each table; and
- tabular numeric styling without using presentation as the semantic signal.

This contract implements the programmatic header relationships required by
WCAG 2.2 Success Criterion 1.3.1. W3C guidance recommends `<th>` plus `scope`
when a data table has both row and column headers; it also warns that applying
header semantics where a cell is not actually a header is itself a structural
failure. Therefore row-header markup is a domain decision, not a blanket first-
column transform.

## Print and export contract

Screen rendering keeps wide tables and canonical JSON keyboard-accessible in
scroll containers. Static media cannot provide that scrolling interaction. The
CSS Overflow Module Level 3 therefore advises authors to adjust print layout so
relevant overflow is simultaneously visible. The shared report stylesheet uses
`@media print` to preserve that distinction: table wrappers and canonical JSON
switch to `overflow: visible`, and the screen-only `32rem` JSON height cap is
removed. Screen overflow behavior remains unchanged.

This matters to the commercial audit surface because the canonical JSON is the
exact-value alternative for reconstruction. Printing or exporting a report to
PDF must not silently replace a complete on-screen audit payload with a clipped
subset.

## Verification

Focused tests exercise complete rendered artifacts and verify that:

1. score-report criterion outcomes start with `<th scope="row">` while evidence-reference cells remain data cells;
2. facets-calibration task, rater, respondent, category/iteration identity axes emit `<th scope="row">`;
3. validation-evidence metric identities emit `<th scope="row">`;
4. canonical JSON in the same artifacts continues to reconstruct the exact governed reports;
5. invalid row-header indices and header/row width drift fail closed; and
6. print media removes scroll clipping and the canonical-JSON height cap while retaining screen overflow behavior.

The tests operate on final standalone artifacts rather than asserting only a
helper substring, so they cover the production composition path and the exact
row-identity distinctions.

## Interpretation boundary

Correct table markup improves programmatic navigation but does not by itself
establish WCAG conformance for the full product, assistive-technology
interoperability across every user agent, psychometric validity, fairness,
reliability, or authorization for consequential scoring. Those claims require
separate evidence.

## References

World Wide Web Consortium. (2023). *Web Content Accessibility Guidelines
(WCAG) 2.2*. https://www.w3.org/TR/WCAG22/

World Wide Web Consortium. (2025, October 7). *CSS Overflow Module Level 3*
(Working Draft). https://www.w3.org/TR/css-overflow-3/

World Wide Web Consortium, Web Accessibility Initiative. (2019, July 27).
*Tables with two headers*. https://www.w3.org/WAI/tutorials/tables/two-headers/

World Wide Web Consortium, Web Accessibility Initiative. (2026, January 26).
*F91: Failure of Success Criterion 1.3.1 for not correctly marking up table
headers*. https://www.w3.org/WAI/WCAG22/Techniques/failures/F91

World Wide Web Consortium, Web Accessibility Initiative. (2026, January 26).
*F46: Failure of Success Criterion 1.3.1 due to using th elements, caption
elements, or non-empty summary attributes in layout tables*.
https://www.w3.org/WAI/WCAG22/Techniques/failures/F46
