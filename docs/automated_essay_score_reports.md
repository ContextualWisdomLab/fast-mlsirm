# Governed automated-essay score reports

## Purpose

`fast_mlsirm.scoring.essay` exposes one content-addressed, source-text-free
`EssayScoreReport` from an exact governed essay request, scoring result, and
engine descriptor. The adapter is a reporting and human-review routing boundary.
It does not score essays, combine analytic criteria, estimate psychometric
parameters, or establish validity.

```python
from fast_mlsirm.scoring.essay import build_essay_score_report

report = build_essay_score_report(
    report_id="essay_score_report",
    request=essay_request,
    result=scoring_result,
    engine=engine_descriptor,
    additional_review_trigger_ids=("scorer_disagreement",),
    metadata={"workflow_stage": "pilot_review"},
)
```

The report embeds the canonical serialized forms and fingerprints of:

- the essay adapter request and its exact task revision;
- the shared governed scoring request;
- the exact engine descriptor;
- the complete shared scoring result and criterion observations; and
- transparent review triggers and audit metadata.

## Standalone HTML audit artifact

The same governed report can be rendered as a self-contained, source-text-free
HTML artifact:

```python
from fast_mlsirm.scoring.essay import render_essay_score_report_html

render_essay_score_report_html(
    report,
    "artifacts/essay_score_report.html",
    title="Pilot Essay Score Audit",
)
```

The renderer first rebuilds the report through the governed request, observation,
result, and report factories and rejects canonical replay differences. The HTML
contains exact report, assessment, rubric, task-revision, engine, request, result,
observation, criterion, trigger, and evidence-reference identities. It does not
contain prompt text, response text, or source text.

The artifact uses semantic landmarks, a keyboard-accessible skip link, labelled
exact-value tables, focusable overflow regions, and a focusable canonical JSON
section. It has no script or external resource dependency. A restrictive
meta-delivered Content Security Policy is included as defense in depth; output
encoding and governed content validation remain required because CSP is not a
replacement for either control.

These implementation choices are informed by WCAG 2.2 and the current Content
Security Policy Level 3 working draft. They support audit usability and reduce
content-injection impact, but they do not constitute a blanket conformance or
security certification. Accessibility conformance still requires full-page,
assistive-technology, and human evaluation in the buyer's deployment context.

## Non-suppressible review triggers

The builder derives structural triggers that callers cannot remove:

- each pre-scoring `EssayReviewFlag` becomes a `submission_*` trigger;
- every abstained, failed, or excluded observation becomes an
  `observation_<status>_<reason>` trigger; and
- a scored observation without an evidence reference becomes
  `observation_missing_evidence`.

Callers may add policy-specific triggers such as scorer disagreement, declared
uncertainty, subgroup warnings, or calibration instability. The report preserves
analytic criteria separately and never averages them into an undeclared holistic
score.

`human_review_required` means only that at least one transparent trigger is
present. A false value means that this structural adapter found no configured or
mandatory trigger. It is not a validity verdict, a deployment authorization, or
evidence that an automated score is interchangeable with a human score.

## Replay and privacy boundary

Before report construction, the adapter validates request, engine, assessment,
rubric, construct, granularity, and criterion provenance for the result and every
observation. It rebuilds nested observations and the scoring result through the
shared governed factories; provenance mismatches, invalid nested state, and
canonical replay differences fail before a report is emitted.

The report stores fingerprints, bounded identifiers, counts, evidence references,
and normalized metadata. It does not store prompt text, response text, or source
text. Sensitive metadata keys remain rejected by the shared scoring contract
safety layer.

## Validity boundary

Operational agreement, fairness, generalizability, robustness, consequences, and
human-scoring comparisons require separate evidence. Content addressing and
review routing support auditability but do not replace a validity argument.
High-stakes use therefore remains subject to an identified human-anchored
validation design, approved adjudication policy, subgroup analysis, monitoring,
and governance.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Shermis, M. D., & Wilson, J. (Eds.). (2024). *The Routledge international
handbook of automated essay evaluation*. Routledge.

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation
and use of automated scoring. *Educational Measurement: Issues and Practice,
31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x

World Wide Web Consortium. (2023, October 5). *Web Content Accessibility
Guidelines (WCAG) 2.2*. https://www.w3.org/TR/WCAG22/

World Wide Web Consortium. (2026, May 5). *Content Security Policy Level 3*
(W3C Working Draft). https://www.w3.org/TR/CSP3/
