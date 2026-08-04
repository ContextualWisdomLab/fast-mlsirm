# Exact task-revision identity in governed scoring

`fast_mlsirm.scoring` separates a task's stable operational name from the exact
content administered in one scoring request. This prevents changed prompts,
forms, or source-bound tasks from being silently pooled under one item-difficulty
parameter.

## Contract

Every schema-1.1 `ScoringRequest` contains:

- `task_id`: a descriptive logical identity;
- `task_family_id`: its operational family; and
- `task_revision_fingerprint`: the complete SHA-256 fingerprint of the exact
  normalized task revision.

The revision is required explicitly. The package does not hash `task_id`, infer
content equality from metadata, or accept an adapter-specific fallback. The
essay adapter projects the complete `EssayPrompt.prompt_fingerprint` into this
field.

Changing only `task_revision_fingerprint` changes `request_fingerprint`. The
logical task and family remain available for reporting, but the revision is the
many-facet estimator item axis.

## Calibration and audit representation

`ScoringFacetsRatingRecord` preserves the logical task, task family, exact task
revision, response revision, request, result, observation, and engine identities.
`ScoringFacetsDesign` discloses aligned arrays:

```text
revision axis: task_revision_fingerprints
logical labels: task_ids
family labels: task_family_ids
response audit: response_task_revision_fingerprints
```

All tensor allocation, duplicate-cell checks, observed support, and
connectedness use the exact revision. One revision cannot be rebound to another
logical task or family. One respondent-revision cell cannot contain different
response artifacts within the bundle occasion.

## Legacy migration

`migrate_scoring_request_v1` accepts only a complete schema-1.0 `to_dict()`
artifact. It:

1. verifies the exact legacy field shape and schema version;
2. replays the authoritative assessment, rubric, score scale, task family, and
   engine authorization;
3. verifies the legacy request fingerprint and public handle;
4. requires an authoritative caller-supplied exact revision; and
5. returns a new schema-1.1 request with a new fingerprint.

Caller metadata is preserved after canonical normalization. Package-managed
engine-policy metadata is validated against and regenerated from the supplied
`AssessmentSpec`. Legacy observations and results are intentionally not migrated
because their provenance remains bound to the old request fingerprint.

## Scientific boundary

A content fingerprint establishes identity, not comparability. Changed task
content may alter difficulty, construct representation, response processes,
subgroup functioning, or rater interpretation. Cross-revision reporting requires
an explicit linking design supported by stable anchors, invariance and DIF
analysis, recovery evidence, uncertainty, and an approved interpretation policy.
When those conditions are absent, revisions remain separate estimator items.

This conservative boundary is consistent with testing standards that require
rationale and evidence before scores from changed forms or procedures are treated
as comparable, and with empirical evidence that prompt characteristics and
anchor drift can affect performance and equating.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Cooperman, A. W., Tai, M. H., DeWeese, J. N., & Weiss, D. J. (2025). Adaptive
measurement of change in the context of item parameter drift. *Applied
Psychological Measurement, 49*(3), 109–125.
https://doi.org/10.1177/01466216241310599

Li, X., & Pan, W. (2025). KAES: Multi-aspect shared knowledge finding and
aligning for cross-prompt automated scoring of essay traits. *Proceedings of the
AAAI Conference on Artificial Intelligence, 39*(23), 24476–24484.
https://doi.org/10.1609/aaai.v39i23.34626

Liu, C., & Jurich, D. (2023). Outlier detection using *t*-test in Rasch IRT
equating under NEAT design. *Applied Psychological Measurement, 47*(1), 34–47.
https://doi.org/10.1177/01466216221124045

Shi, B., Huang, L., & Lu, X. (2020). Effect of prompt type on test-takers’
writing performance and strategy use. *Language Testing, 37*(3), 361–388.
https://doi.org/10.1177/0265532220911626
