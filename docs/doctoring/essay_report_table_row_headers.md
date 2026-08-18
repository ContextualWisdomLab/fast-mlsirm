# Essay report table row-header accessibility

## Decision

Standalone essay calibration and validation reports mark the first identity or ordered-axis cell in each data row as an HTML table header with `scope="row"`. Column headings remain `th scope="col"`. This preserves the report's existing values and visual layout while making row/data relationships programmatically determinable for assistive technology.

The rule applies only when the first column actually identifies the row. It does not authorize semantic header markup for layout tables or arbitrary presentation-only cells.

## Verification contract

Regression tests render the public calibration and validation HTML artifacts and require the expected task, rater, respondent, threshold/iteration, and metric identities to appear as `th scope="row"` cells. The implementation reuses the shared `_table` renderer, which validates the selected header-column index against the declared table width.

## Standards basis

WCAG 2.2 Success Criterion 1.3.1 requires information, structure, and relationships conveyed through presentation to be programmatically determinable or available in text. W3C Technique H63 describes `scope="row"` and `scope="col"` as mechanisms for associating table header cells with their data cells; Technique H51 describes semantic table markup for tabular information. W3C Failure F91 identifies missing programmatically determinable table headers as a failure pattern for Success Criterion 1.3.1.

These techniques are informative implementation guidance rather than a claim that this isolated change establishes whole-product WCAG conformance.

## References — APA 7

World Wide Web Consortium. (2024, December 12). *Web Content Accessibility Guidelines (WCAG) 2.2*. https://www.w3.org/TR/WCAG22/

World Wide Web Consortium. (2024, September 13). *H63: Using the scope attribute to associate header cells with data cells in data tables*. https://www.w3.org/WAI/WCAG22/Techniques/html/H63

World Wide Web Consortium. (2026, January 12). *H51: Using table markup to present tabular information*. https://www.w3.org/WAI/WCAG22/Techniques/html/H51

World Wide Web Consortium. (2026, January 26). *F91: Failure of Success Criterion 1.3.1 for not correctly marking up table headers*. https://www.w3.org/WAI/WCAG22/Techniques/failures/F91
