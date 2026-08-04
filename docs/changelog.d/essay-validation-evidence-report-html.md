# Accessible standalone essay validation evidence artifacts

## Added

- Added `render_essay_validation_evidence_report_html`, which revalidates one governed `EssayValidationEvidenceReport` and emits a deterministic, source-text-free, script-free standalone HTML audit artifact.
- The artifact exposes exact report, assessment, construct, rubric, validation-dataset, automated-engine, human-reference, metric, review-trigger, and interpretation-boundary values through semantic landmarks, keyboard-accessible exact-value tables, and canonical JSON.
- A restrictive meta-delivered Content Security Policy and output encoding reduce injection impact; the artifact deliberately excludes score-label vectors, universal thresholds, Boolean pass fields, validity, fairness, model-selection claims, and deployment authorization.
