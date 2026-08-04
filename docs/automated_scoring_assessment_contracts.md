# Automated-Scoring Assessment Contracts

`fast_mlsirm.scoring` provides the shared provider-neutral contract boundary used by future essay-scoring, RAG-evaluation, and enterprise-issue verticals. It prevents each vertical from defining a different rubric identity, missing-value policy, model policy, or audit representation.

## Contract flow

```text
RubricSpecification[]
        +
ConstructSpec[]
        +
PolicyDocument[]
        ↓
build_assessment_spec(...)
        ↓
AssessmentSpec
```

`fast_mlsirm.rubric.RubricSpecification` remains the sole rubric source of truth. An assessment binds the exact rubric ID, rubric semantic version, complete SHA-256 fingerprint, construct ID, and response format. It does not copy or reinterpret score levels.

## Example

```python
from fast_mlsirm.rubric import (
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
)
from fast_mlsirm.scoring import (
    ConstructSpec,
    PolicyKind,
    build_assessment_spec,
    build_policy_document,
)

rubric = RubricSpecification(
    rubric_id="evidence_quality",
    construct_id="evidence_quality",
    construct_definition="Evidence-conditioned answer quality.",
    response_format=ResponseFormat.ORDINAL_RATING,
    levels=(
        RubricLevel(
            score=0,
            label="not_supported",
            descriptor="The response is not supported by the admitted evidence.",
            observable_indicators=("a material claim lacks admitted evidence",),
        ),
        RubricLevel(
            score=1,
            label="fully_supported",
            descriptor="Every material claim is supported by admitted evidence.",
            observable_indicators=("each material claim maps to evidence",),
        ),
    ),
    task_families=("evidence_review",),
    evidence_requirements=("retain exact evidence provenance",),
)

construct = ConstructSpec(
    construct_id="evidence_quality",
    label="Evidence quality",
    definition="Quality of a response after conditioning on admitted evidence.",
)

policies = tuple(
    build_policy_document(
        policy_id=f"{policy_kind.value}_document",
        policy_version="1.0.0",
        policy_kind=policy_kind,
        settings={"policy_mode": "strict"},
    )
    for policy_kind in PolicyKind
)

assessment = build_assessment_spec(
    assessment_id="evidence_assessment",
    assessment_version="1.0.0",
    constructs=(construct,),
    rubrics=(rubric,),
    policy_documents=policies,
    metadata={"deployment_stage": "pilot"},
)

print(assessment.assessment_handle)
print(assessment.rubric_fingerprints)
```

## Required policy families

Every assessment has exactly one content-addressed policy document for each family:

- `engine_policy`
- `calibration_policy`
- `validation_policy`
- `adjudication_policy`
- `monitoring_policy`

The contracts do not invent default statistical thresholds. Policy settings are bounded JSON that later orchestration modules interpret through separately reviewed behavior.

## Trust and privacy boundaries

- Factory seals are API-governance controls, not cryptographic authorization inside a hostile Python process.
- SHA-256 fingerprints identify exact normalized content; they are not signatures, credentials, or permissions.
- Metadata rejects raw response, essay, prompt, provider-output, and source-content fields.
- Errors expose bounded machine-readable codes and JSON-style paths without rejected response text.
- A valid contract does not establish scoring accuracy, reliability, fairness, model fit, scoreability, or validity.
- Numerical estimation remains in the existing Rust/PyO3 layer. This module performs no likelihood, gradient, uncertainty, DIF, linking, or utility arithmetic.

## Modular deployment

The contracts are ordinary immutable Python values with deterministic JSON representations. A standalone caller may use them in-process. An MSA deployment may serialize the same values at a service boundary while retaining the full fingerprints in events, model registries, audit stores, and downstream scoring results.

No persistence, tenancy, authentication, provider SDK, or network call is introduced by this slice.

## Methodological basis

The design treats automated scoring as an assessment system rather than a one-shot prediction function. Rubric revisions, human and automated raters, calibration, validation, adjudication, monitoring, and audit evidence must remain traceable across the complete score lifecycle.

### References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
