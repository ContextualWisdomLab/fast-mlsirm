# Doctoring record: diagnostics-report focus reveal and contrast preservation

## Decision

The standalone diagnostics report reveals its visually hidden skip-navigation link whenever that link has actual input focus. The stylesheet keeps both `.skip-link:focus` and `.skip-link:focus-visible`, using the same visible placement and outline treatment.

Pointer hover may highlight the active bar or table row, but it no longer lowers the opacity of every unrelated row. Data that remains active and readable should retain its normal foreground and background colors while another row is hovered.

The report suppresses the browser's pointer-triggered focus outline only for the scrollable table wrapper, exact-value and export summaries, and the scrollable export block. The suppression selector is narrowly written as `:focus:not(:focus-visible)`. A bare `:focus { outline: none; }` rule is prohibited. Each affected component retains a repository-owned `:focus-visible` indicator with an explicit three-pixel outline.

## Standards rationale

Selectors Level 4 distinguishes `:focus`, which applies while an element has input focus, from `:focus-visible`, which additionally depends on user-agent heuristics about whether a focus indicator should be drawn. A skip link that is visually hidden until focused therefore uses `:focus` as the reveal boundary rather than relying only on the heuristic pseudo-class.

WCAG 2.2 Success Criterion 1.4.3 requires minimum text contrast for ordinary visible text, subject to its stated exceptions. Applying opacity to an entire otherwise active row composites both text and background and can reduce the final rendered contrast. Because the actual ratio depends on all computed colors and the rendering environment, this change does not claim that the removed selector was universally nonconforming or that the resulting report is formally WCAG-conformant. It removes an avoidable contrast risk and retains exact visible content for all rows.

WCAG Technique C45 describes `:focus-visible` as a way to provide a keyboard focus indicator without necessarily presenting the same author style after pointer interaction. It also notes that pointer users can benefit from visible focus in some circumstances. The report therefore treats pointer-only outline suppression as bounded presentation polish rather than a general rule: it applies only to the named report containers, preserves the strong keyboard indicator, and does not suppress focus on form controls or links.

The `:focus:not(:focus-visible)` form is also a compatibility boundary. A user agent that does not understand `:focus-visible` rejects that selector instead of applying a bare outline removal, leaving the user-agent focus presentation available. This is a progressive-enhancement property, not a claim of browser-wide or assistive-technology conformance.

## Verification contract

`tests/test_report_focus_contrast.py` renders a realistic diagnostics report through the public `render_diagnostics_report` entry point and proves that:

- the generated stylesheet includes a `.skip-link:focus` reveal rule alongside `.skip-link:focus-visible`;
- the focused link retains the strong repository-owned outline;
- the generated stylesheet contains no chart-peer opacity selector;
- the generated stylesheet contains no table-peer opacity selector;
- no affected component uses a bare `:focus` outline-removal rule;
- the table wrapper, report summaries, and export block suppress only `:focus:not(:focus-visible)`; and
- every affected component retains its explicit three-pixel `:focus-visible` outline.

The test validates deterministic generated markup. It is not a substitute for browser, assistive-technology, zoom, forced-colors, dark-mode, print, pointer-modality, or user testing.

## Compatibility and rollback

The change affects presentation only. It does not alter report data, JSON parsing, numerical values, table semantics, export formats, model fitting, database objects, network behavior, or public Python signatures. Rollback consists of restoring the prior CSS selectors, but doing so would reintroduce the documented focus-reveal and contrast risks. A rollback must not replace the bounded selector with a bare `:focus { outline: none; }` rule.

## References

World Wide Web Consortium. (2023, October 5). *Web Content Accessibility Guidelines (WCAG) 2.2* (W3C Recommendation). https://www.w3.org/TR/WCAG22/

World Wide Web Consortium Accessibility Guidelines Working Group. (2025, September 25). *C45: Using CSS `:focus-visible` to provide keyboard focus indication*. https://www.w3.org/WAI/WCAG22/Techniques/css/C45

World Wide Web Consortium Accessibility Guidelines Working Group. (2025). *Understanding Success Criterion 2.4.7: Focus visible*. https://www.w3.org/WAI/WCAG22/Understanding/focus-visible.html

World Wide Web Consortium CSS Working Group. (2026, January 22). *Selectors Level 4* (W3C Working Draft). https://www.w3.org/TR/selectors-4/
