# Exact task-revision identity in governed scoring

## Purpose

A logical task identifier is an administrative label, not evidence that two task
forms contain the same content or support interchangeable score interpretations.
Scoring-request wire schema `1.1` therefore requires a complete lowercase
SHA-256 `task_revision_fingerprint` for the exact normalized task content.

The shared contract separates:

- `task_id`: stable logical label for display and administration;
- `task_family_id`: declared family used by the rubric and assessment;
- `task_revision_fingerprint`: exact content identity used by calibration.

This separation applies to standalone scoring, domain adapters, serialized MSA
events, and the criterion-level many-facet handoff. It prevents two changed
prompts or tasks from being silently pooled under one item-difficulty parameter.

## Request schema 1.1

Every current request factory requires the revision explicitly:

```python
from fast_mlsirm.scoring import build_scoring_request

request = build_scoring_request(
    request_id="request_identity",
    assessment=assessment_spec,
    rubric=rubric_specification,
    granularity="criterion_level",
    respondent_id="respondent_identity",
    response_id="response_identity",
    task_id="logical_task_identity",
    task_revision_fingerprint=task_sha256,
    task_family_id="declared_task_family",
    occasion_id="initial_occasion",
    criterion_ids=("claim_support", "source_alignment"),
    response_content_fingerprint=response_sha256,
    response_character_count=128,
    response_unit_count=8,
)
```

The revision is canonical request content and therefore changes
`request_fingerprint`. Callers must compute it from an authoritative normalized
task representation. The package does not derive a revision from `task_id`, an
adapter metadata key, a provider name, or a response.

The essay adapter uses the complete `EssayPrompt.prompt_fingerprint` as the
shared task revision. Adapter metadata may repeat that identity for domain audit,
but the shared request field is authoritative for calibration.

## Legacy schema 1.0 migration

A schema-`1.0` request did not contain exact task content identity. Migration is
therefore explicit and requires an authoritative caller-supplied revision:

```python
from fast_mlsirm.scoring import migrate_scoring_request_v1

current_request = migrate_scoring_request_v1(
    legacy_request_artifact,
    assessment=authoritative_assessment,
    rubric=authoritative_rubric,
    task_revision_fingerprint=task_sha256,
)
```

The migration boundary:

1. requires the exact schema-`1.0` field set;
2. verifies the authoritative assessment, rubric, task family, score scale, and
   engine-policy projection through current contracts;
3. preserves normalized caller metadata while replacing package-managed
   authorization metadata from the authoritative assessment;
4. reconstructs schema-`1.0` canonical content and verifies its fingerprint and
   public handle; and
5. emits a new schema-`1.1` request with a new request fingerprint.

Observations and results are not migrated. They remain bound to the legacy
request fingerprint and must be produced again under the current request.

## Calibration identity

The many-facet estimator receives a respondent-by-task-revision-by-rater tensor.
Each `ScoringFacetsDesign` retains aligned logical labels:

```text
task_revision_fingerprints  -> estimator item axis
task_ids                    -> aligned logical labels
task_family_ids             -> aligned family labels
```

One revision may map to exactly one `(task_id, task_family_id)` pair. One logical
task may have multiple revisions. Duplicate cells, observed-support checks,
respondent–item connectedness, item–rater connectedness, dense-allocation
bounds, response binding, design replay, and bundle replay all use the exact
revision identity.

## Comparability and linking boundary

A content fingerprint establishes identity, not equivalence. Changed task
content may change difficulty, construct representation, subgroup functioning,
strategy use, or rater interpretation. Two task revisions remain separate
calibration items unless an approved linking design supplies evidence such as:

- common-item or common-person anchors with documented quality;
- stable anchor parameters and drift monitoring;
- measurement-invariance and differential-item-functioning analyses;
- true-parameter and linking-error recovery studies; and
- an explicit policy governing permitted score interpretations and uses.

The package therefore fails closed on accidental pooling and makes no automatic
cross-revision linking claim.

## Operational guidance

- Store the complete revision fingerprint in event envelopes, audit stores,
  model registries, and scoring-result provenance.
- Treat every task-content change as a new revision, including changes to
  instructions, source context, response constraints, scoring-relevant media,
  or prompt wording.
- Preserve logical IDs for reporting, but never use them as estimator equality
  keys.
- Re-execute legacy observations after migration.
- Quarantine a revision when anchor drift, DIF, or construct-representation
  evidence invalidates the approved linking policy.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking:
Methods and practices* (3rd ed.). Springer.
https://doi.org/10.1007/978-1-4939-0317-7

Li, Y. (2012). Examining the impact of drifted polytomous anchor items on test
characteristic curve linking and IRT true score equating (Research Report No.
RR-12-09). *ETS Research Report Series, 2012*(1), i–21.
https://doi.org/10.1002/j.2333-8504.2012.tb02291.x

Millsap, R. E. (2011). *Statistical approaches to measurement invariance*.
Routledge. https://doi.org/10.4324/9780203821961

Shi, B., Huang, L., & Lu, X. (2020). Effect of prompt type on test-takers’
writing performance and strategy use. *Language Testing, 37*(3), 361–388.
https://doi.org/10.1177/0265532220911626
