# Accessible standalone essay facets-calibration artifacts

## Added

- Added `render_essay_facets_calibration_report_html`, which replay-verifies one governed `EssayFacetsCalibrationReport` and emits a deterministic, source-text-free, script-free standalone HTML audit artifact.
- The artifact exposes exact report, design, assessment, rubric, construct, occasion, criterion, respondent, task-revision, rater-engine, category, estimate, convergence, connectedness, iteration, and review-trigger evidence through semantic landmarks, keyboard-accessible exact-value tables, and canonical JSON.
- A restrictive meta-delivered Content Security Policy and output encoding reduce injection impact; convergence and connectedness remain integrity prerequisites and do not establish model fit, reliability, fairness, scorer interchangeability, construct validity, global optimality, or deployment authorization.
- Confined report publication to a canonical caller-approved output directory, with current-working-directory defaults, traversal and absolute-escape rejection, existing symlink-parent resolution, post-creation parent revalidation, and fail-closed non-directory roots.