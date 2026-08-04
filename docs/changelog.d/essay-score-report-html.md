# Accessible standalone essay score report artifacts

## Added

- Added `render_essay_score_report_html`, which replay-verifies one governed `EssayScoreReport` and emits a deterministic, source-text-free, script-free standalone HTML audit artifact.
- The artifact exposes exact report, assessment, rubric, task-revision, engine, request, result, observation, criterion, trigger, and evidence-reference identities through semantic landmarks, keyboard-accessible exact-value tables, and canonical JSON.
- Empty states use explicit status semantics, numeric table cells use tabular numerals, skip links remain visible for any received focus, and motion-sensitive users receive a reduced-motion override without dimming non-hovered report rows.
- A restrictive meta-delivered Content Security Policy and output encoding reduce content-injection impact. Review routing remains an audit signal only and does not establish scoring validity, fairness, reliability, interchangeability, accessibility conformance, security certification, or authorization for consequential deployment.
