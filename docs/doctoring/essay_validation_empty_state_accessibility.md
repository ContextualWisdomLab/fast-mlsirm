# Essay-validation empty-state accessibility doctoring

## Scope

This record governs the empty identifier state emitted by the standalone automated-essay validation evidence report. The report is serialized, script-free HTML. The change is intentionally semantic only: it does not alter psychometric evidence, scoring, validation thresholds, focus order, or consequential-decision policy.

## Standards decision

The empty-state container uses `role="status"` because the message is advisory status information and is not important enough to justify an alert. WAI-ARIA defines `status` as a live-region role with implicit `aria-live="polite"` and `aria-atomic="true"`, and advises that a status should not receive focus as a consequence of a status change. We additionally serialize `aria-atomic="true"` explicitly so the intended whole-message atomicity remains obvious and interoperable to readers and tooling.

The renderer does **not** claim that opening a pre-populated static HTML file will cause the empty-state message to be spoken automatically. The June 2026 WAI-ARIA 1.3 Working Draft clarifies that assistive technology typically conveys changes to live regions rather than their initial contents. Consumers that later hydrate or dynamically update this report may benefit from the live-region semantics, while the static artifact remains visibly readable and available in the accessibility tree without forced focus movement.

WAI-ARIA 1.2 remains the current W3C Recommendation; WAI-ARIA 1.3 is recorded as the newest Working Draft consulted for the clarified live-region creation behavior. No draft-only feature is required by this implementation.

## Verification contract

`tests/test_scoring_essay_validation_report_html.py` requires the exact empty-state markup to contain both `role="status"` and explicit `aria-atomic="true"`. Existing deterministic rendering, escaping, CSP, script-free output, provenance, and scientific-boundary tests remain unchanged. The accessibility claim is deliberately bounded to semantic exposure and update behavior; it is not a cross-screen-reader announcement guarantee.

## References

World Wide Web Consortium. (2023, June 6). *Accessible Rich Internet Applications (WAI-ARIA) 1.2* (W3C Recommendation). https://www.w3.org/TR/wai-aria-1.2/

World Wide Web Consortium. (2026, June 4). *Accessible Rich Internet Applications (WAI-ARIA) 1.3* (W3C Working Draft). https://www.w3.org/TR/2026/WD-wai-aria-1.3-20260604/

World Wide Web Consortium. (n.d.). *ARIA22: Using role=status to present status messages*. Web Accessibility Initiative. https://www.w3.org/WAI/WCAG21/Techniques/aria/ARIA22.html
