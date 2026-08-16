# Doctoring record: diagnostics-report focus reveal and contrast preservation

## Decision

The standalone diagnostics report reveals its visually hidden skip-navigation link whenever that link has actual input focus. The stylesheet keeps both `.skip-link:focus` and `.skip-link:focus-visible`, using the same visible placement and outline treatment.

Pointer hover may highlight the active bar or table row, but it does not lower the opacity of unrelated rows. Data that remains active and readable retains its normal foreground and background colors while another row is hovered.

For keyboard-focusable generic scroll and disclosure containers, pointer-triggered browser outlines are suppressed only through the bounded selector `:focus:not(:focus-visible)`. The report retains explicit three-pixel `:focus-visible` indicators on the table wrapper, exact-value and export summaries, and the focusable export block. Bare `:focus { outline: none; }` rules are prohibited.

## Standards rationale

Selectors Level 4 distinguishes `:focus`, which applies while an element has input focus, from `:focus-visible`, which additionally depends on user-agent heuristics about whether a focus indicator should be drawn. A skip link that is visually hidden until focused therefore uses `:focus` as the reveal boundary rather than relying only on the heuristic pseudo-class.

Selectors Level 4 also requires an unsupported pseudo-class to make its selector invalid and therefore match nothing. Consequently, a user agent without usable `:focus-visible` support ignores the entire bounded `:focus:not(:focus-visible)` rule instead of applying an unconditional outline removal. Its native fallback focus indication remains available.

W3C Technique C45 documents `:focus-visible` as a sufficient technique only when the resulting keyboard-focus indicator is visible. It also notes that pointer users may benefit from explicit focus indication. This report therefore treats pointer-only outline suppression as narrow presentation polish rather than a universal accessibility rule and never removes the keyboard-focused indicator.

WCAG 2.2 Success Criterion 2.4.7 requires a mode of operation where keyboard focus is visible. The report's authored three-pixel indicators support that objective. This source-level contract does not establish Focus Appearance at Level AAA, because rendered area and contrast depend on the final browser, theme, zoom, and surrounding pixels.

WCAG 2.2 Success Criterion 1.4.3 requires minimum text contrast for ordinary visible text, subject to its stated exceptions. Applying opacity to an entire otherwise active row composites both text and background and can reduce the final rendered contrast. Because the actual ratio depends on all computed colors and the rendering environment, this change does not claim that the removed selector was universally nonconforming or that the resulting report is formally WCAG-conformant. It removes an avoidable contrast risk and retains exact visible content for all rows.

## Verification contract

`tests/test_report_focus_contrast.py` renders a realistic diagnostics report through the public `render_diagnostics_report` entry point and proves that:

- the generated stylesheet includes a `.skip-link:focus` reveal rule alongside `.skip-link:focus-visible`;
- the focused skip link retains the strong repository-owned outline;
- the generated stylesheet contains no chart-peer or table-peer opacity selector;
- no bare table, summary, or export-block `:focus { outline: none; }` rule is emitted;
- each intended generic container uses `:focus:not(:focus-visible)` for bounded pointer suppression; and
- every affected keyboard path keeps its explicit three-pixel `:focus-visible` indicator.

The existing report regression also verifies the table wrapper's bounded selector. These tests execute the report generator and inspect deterministic generated markup; they are not substitutes for browser, assistive-technology, pointer-modality, zoom, forced-colors, dark-mode, print, or user testing.

## Compatibility and rollback

The change affects presentation only. It does not alter report data, JSON parsing, numerical values, table semantics, export formats, model fitting, database objects, network behavior, or public Python signatures.

Rollback must restore the CSS generator, both regression modules, CHANGELOG entry, and this doctoring record together. Restoring a bare `:focus { outline: none; }` rule would reopen the unsupported-selector fallback defect; removing `:focus-visible` indicators would violate the documented keyboard-focus contract.

## References

World Wide Web Consortium. (2023, October 5). *Web Content Accessibility Guidelines (WCAG) 2.2* (W3C Recommendation). https://www.w3.org/TR/WCAG22/

World Wide Web Consortium. (2025, September 17). *Understanding Success Criterion 2.4.7: Focus visible*. https://www.w3.org/WAI/WCAG22/Understanding/focus-visible

World Wide Web Consortium. (2025, September 25). *C45: Using CSS `:focus-visible` to provide keyboard focus indication*. https://www.w3.org/WAI/WCAG22/Techniques/css/C45

World Wide Web Consortium CSS Working Group. (2026, January 22). *Selectors Level 4* (W3C Working Draft). https://www.w3.org/TR/selectors-4/
