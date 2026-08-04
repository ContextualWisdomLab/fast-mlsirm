# Task-Revision Calibration Identity Design

## Problem

A shared `ScoringRequest` currently exposes a logical `task_id` but no provider-neutral identity for the exact task content revision. The essay adapter retains `EssayPrompt.prompt_fingerprint` only in domain metadata. A calibration handoff that uses `task_id` as the item axis can therefore pool changed prompt content under one item difficulty without an explicit linking decision.

The product must make accidental cross-revision equality impossible while retaining logical task identifiers for reporting and administration.

## Decision

Add a required SHA-256 `task_revision_fingerprint` to the shared scoring request and make the exact revision—not the logical task ID—the default estimator item identity.

The request wire contract advances independently from `1.0` to `1.1`. Assessment, observation, result, and policy schema versions remain unchanged. Existing `1.0` request artifacts are migrated only through an explicit, fail-closed helper that requires the missing revision fingerprint from an authoritative caller. The package never derives a revision from `task_id` or an adapter-specific metadata key.

## Public contracts

### Scoring request

`ScoringRequest` gains:

```python
task_revision_fingerprint: str
```

The value is required by both execution-layer and authorization-layer factories, validated as a complete lowercase SHA-256 digest, serialized in canonical request content, and included in `request_fingerprint`.

New constants are explicit package attributes:

```python
SCORING_REQUEST_SCHEMA_VERSION = "1.1"
LEGACY_SCORING_REQUEST_SCHEMA_VERSION = "1.0"
```

They are not added to the pinned star-import surface in this release.

### Legacy migration

`migrate_scoring_request_v1(...)` accepts a complete legacy `to_dict()` artifact, the authoritative assessment and rubric, and an explicit `task_revision_fingerprint`.

The migration:

1. requires an exact bounded mapping with schema `1.0`;
2. validates the legacy assessment, rubric, response format, granularity, task family, score scale, identifiers, counts, metadata, and authorization projection through current factories;
3. reconstructs the old canonical content without a revision field and verifies the legacy request fingerprint and handle;
4. returns a new schema `1.1` request whose fingerprint changes;
5. does not migrate observations or results, because they remain bound to the old request fingerprint and require re-execution.

No essay-specific fallback is accepted.

## Domain adapters

`build_essay_scoring_request` supplies `prompt.prompt_fingerprint` as the shared `task_revision_fingerprint`. It may retain `essay_prompt_fingerprint` in adapter metadata for domain audit, but the shared field is authoritative for calibration identity.

Changing any normalized `EssayPrompt` content while retaining `prompt_id` creates a distinct shared request and a distinct calibration item.

## Calibration model

`ScoringFacetsRatingRecord` retains both display and estimator identities:

- `task_id`: logical descriptive task identifier;
- `task_family_id`: logical task family;
- `task_revision_fingerprint`: exact content revision and estimator item identity.

`ScoringFacetsDesign` exposes aligned item metadata:

- `task_revision_fingerprints`: unique sorted estimator axis;
- `task_ids`: logical IDs aligned one-for-one with revisions; duplicates are allowed;
- `task_family_ids`: aligned task families;
- `response_task_revision_fingerprints`: exact revision for every response audit entry.

Tensor cells, duplicate detection, observed-support checks, respondent–task connectedness, task–rater connectedness, and dense allocation all use `(respondent_id, task_revision_fingerprint, engine_fingerprint)`. Logical IDs remain reportable but never silently collapse revisions.

One revision fingerprint must map to exactly one `(task_id, task_family_id)` pair. One logical task may map to multiple revisions. One respondent-revision cell may map to only one exact response revision within an occasion.

## Linking policy boundary

This slice implements the conservative default: one exact task revision equals one estimator item. It does not implement common-item, common-person, or chained linking across revisions.

Any future linking policy must be explicit, content-addressed, and supported by anchor quality, drift, DIF/invariance, recovery, and comparability evidence. A digest prevents accidental equality; it does not prove equivalent difficulty or construct representation.

## Error handling

All failures remain bounded, structured, and non-reflective. New stable error families include:

- `invalid_task_revision_fingerprint`;
- `task_revision_provenance_conflict`;
- `respondent_task_revision_response_conflict`;
- `insufficient_facets_task_revisions`;
- `invalid_legacy_scoring_request`;
- `legacy_request_fingerprint_mismatch`;
- `legacy_request_handle_mismatch`.

## Testing

Tests cover:

- required and malformed shared revision identities;
- canonical request schema `1.1` and changed request fingerprints;
- exact essay prompt revision propagation;
- same logical prompt ID with changed content producing separate item-axis entries;
- logical ID retention in calibration reports;
- conflicting revision-to-logical provenance rejection;
- response and bundle replay after post-construction mutation;
- explicit `1.0` migration, tamper rejection, and old-result non-rebinding;
- focused and full Python suites, Rust workspace, PyO3, GPU-no-skip, security, SAST, docstring, and coverage gates.

## Documentation and release

Add a buyer-facing contract guide and an authoritative changelog fragment, then render tracked `CHANGELOG.md`. This closes issue #499 but does not by itself make the broader automated-essay vertical release-ready.

## Evidence

The design follows the requirement that changed testing procedures need evidence before score comparability is claimed, that interchangeable forms require rationale and statistical evidence, and that anchor characteristics must be documented. Empirical work also shows that prompt characteristics can affect writing performance and that drifted anchors can bias linking/equating.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Li, Y. (2012). Examining the impact of drifted polytomous anchor items on test characteristic curve (TCC) linking and IRT true score equating (Research Report No. RR-12-09). Educational Testing Service. https://doi.org/10.1002/j.2333-8504.2012.tb02291.x

Qu, Y., Wei, Y., & Morgan, R. (2016). *The impact of population shift on equating and differential item functioning of anchor items: An empirical study* (Research Memorandum No. RM-16-05). Educational Testing Service.

Shi, B., Huang, L., & Lu, X. (2020). Effect of prompt type on test-takers’ writing performance and strategy use. *Language Testing, 37*(3), 361–388. https://doi.org/10.1177/0265532220911626

Zou, Y., Kannan, S., & Sidhu, G. K. (2024). Influence of task complexity on text features and writing scores. *SAGE Open, 14*(3). https://doi.org/10.1177/21582440241284186
