# Governed automated-essay scoring adapters

`fast_mlsirm.scoring.essay` is the first domain adapter over the shared
provider-neutral scoring contracts. It records prompt, submission, evidence-span,
and request provenance without introducing a second rubric, observation, result,
or engine schema.

## Boundary

The adapter stores no prompt text, essay text, external source text, generated
feedback, credentials, or provider output. Callers retain content in an
appropriately governed system and pass exact SHA-256 fingerprints, descriptive
identifiers, bounded counts, and source-text-free `EvidenceReference` values.

A valid adapter proves structural integrity and provenance only. It does not prove
writing quality, scoring accuracy, rater interchangeability, construct validity,
fairness, reliability, scoreability, or readiness for consequential use.

## Workflow

```text
EssayPrompt
    + EssaySubmission
    + EssayResponseEvidence[]
    + AssessmentSpec
    + RubricSpecification
                |
                v
       EssayScoringRequest
                |
                v
     shared ScoringRequest
                |
                v
        shared ScoringEngine
                |
                v
        shared ScoringResult
```

`build_essay_scoring_request` always creates a criterion-level shared request. It
binds the exact assessment and rubric revisions, prompt and submission
fingerprints, response-content fingerprint and counts, declared task family,
criterion identifiers, review signals, and evidence-adapter fingerprints. Score
categories still come only from the bound rubric.

`score_essay_request` calls the existing runtime-checkable `ScoringEngine`
protocol and rejects a result whose request or engine identity differs from the
adapter and descriptor. Deterministic examples can use the existing
`StaticFixtureEngine`; human and automated fixtures therefore exercise the same
observation and result factories that later providers must use.

## Review signals

`EssayReviewFlag` distinguishes pre-scoring review signals such as malformed or
off-topic responses, prompt-copying risk, adversarial responses, low evidence
coverage, and surface-feature shortcut risk. A flag is not a score and cannot
silently change a rubric category. Later adjudication and validation layers may
use these exact signals under an explicit policy.

## Evidence spans

`EssayResponseEvidence` wraps, rather than replaces, the common
`EvidenceReference`. It adds prompt/submission provenance, a span kind, and
bounded half-open character offsets. Response spans must reference the exact
response identity; prompt spans must reference the exact prompt identity.
External-source spans retain their own source identity. Evidence text is never
stored in the adapter.

## Scientific sequence

This slice intentionally adds no estimator. The next development layer should
marshal criterion observations into the existing Rust-backed many-facet ordinal
calibration path and verify connectedness and true-parameter recovery. Rater
severity, prompt difficulty, range restriction, criterion bias, drift, DIF,
multidimensional or bifactor structure, testlets, and latent-space residual
interactions must be introduced only in that evidence-driven sequence. Human
score correlation alone is not a validity argument.

## Security and privacy

- all public identities use descriptive two-or-more-token `snake_case` values;
- full SHA-256 fingerprints are authoritative; 128-bit handles are display and
  lookup aids rather than signatures;
- metadata is bounded, recursively copied, immutable, and rejects sensitive
  response/prompt/source content fields;
- direct construction of prompt, submission, evidence, and request adapters is
  rejected in favor of validating factories;
- prompt/submission/evidence replay mismatches fail closed with stable,
  non-reflective errors;
- no provider SDK, network request, raw content logging, or numerical scoring is
  added.

## Evidence basis

The implementation follows the validation boundary described in automated
scoring practice: automated scores require evidence about construct
representation, human and automated rater behavior, generalization, fairness,
and operational consequences rather than correlation alone. The adapter is an
infrastructure prerequisite for those studies, not a substitute for them.

### References

Shermis, M. D., & Wilson, J. (Eds.). (2024). *The Routledge international
handbook of automated essay evaluation*. Routledge.

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation
and use of automated scoring. *Educational Measurement: Issues and Practice,
31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
