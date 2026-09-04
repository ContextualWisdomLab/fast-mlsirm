# Governed Rater Measurement Context Map

Status: **Proposed DDD baseline**  
Date: **2026-08-29**  
Canonical published language: `cwl_governed_rater_observation/v1`

## Purpose

This document defines the strategic and tactical Domain-Driven Design boundary
for governed human, model, and algorithmic raters. It deliberately excludes
CEFR, a particular instrument, a particular decision policy, and publication or
paper-delivery concerns. Domain profiles may be added later without changing the
core bounded contexts.

The core invariant is:

> An observation is not a score, and a score is not a decision.

## Ubiquitous language

| Term | Meaning |
|---|---|
| Rater family | A stable human, model, or algorithm lineage. |
| Rater configuration | One exact reusable configuration: provider or employing authority, implementation revision, instruction revision, response schema, workflow mode, and modality. |
| Rater invocation | One execution of one rater configuration. Repeated invocations are nested observations, not new independent raters. |
| Criterion observation | An ordered-category anchor with evidence, or an explicit abstention. |
| Observation panel | A governed collection of invocations designed for calibration or scoring. |
| Calibration design | The immutable identification and linking design used by the numerical context. |
| Parameter snapshot | An immutable numerical artifact for severity, thresholds, discrimination, interactions, and uncertainty. |
| Score snapshot | A product-owned immutable publication of a numerical result reference and its limitations. |
| Decision policy | A separately versioned policy that consumes score uncertainty and consequences. |
| Monitoring artifact | A temporal result that distinguishes invocation noise, gradual drift, and configuration change. |

## Bounded contexts

### 1. Rater Observation — `contextual-orchestrator`

**Aggregate root:** `RaterInvocation`

Owns provider-neutral execution, independent blinding, structured-output
validation, sanitized usage, failure and abstention preservation, and evidence
references. It emits criterion observations only.

It must not own latent scores, cut scores, placement, certification, instrument
publication, human-review persistence, or rater-parameter estimation.

### 2. Measurement Calibration — `fast-mlsirm`

**Aggregate roots:** `CalibrationDesign`, `ObservationPanel`,
`ParameterSnapshot`

Owns the published observation language and all production psychometric
arithmetic: rater severity, category thresholds, discrimination and range
compression, task and criterion interactions, invocation-level variance,
differential rater functioning, uncertainty, recovery, and CPU/GPU parity.

It must not own source responses, provider calls, participant consent, panel
workflow state, or business decisions.

### 3. Assessment Operations — `psychometrics-commons`

**Aggregate roots:** `RaterPanel`, `ObservationRequest`, `AdjudicationCase`,
`ScorePublication`

Owns assignments, blind-panel lifecycle, anchor allocation, observation receipt,
scoring dispatch, adjudication, immutable result publication, consent, and
product audit state.

It consumes numerical artifacts through released contracts. It must not
reimplement estimation or mutate external rater invocations.

### 4. Temporal Measurement Monitoring — `TEPP`

**Aggregate roots:** `RaterMonitoringRun`, `MonitoringArtifact`

Owns leakage-safe multi-clock monitoring, time-varying parameter analysis,
gradual drift, change-point evidence, cross-classified memberships, and
longitudinal invariance artifacts.

It must not rewrite a published score, reinterpret an invocation as a different
configuration, or infer an operational decision.

### 5. Measurement Context Registry — `semantic-data-portal`

**Aggregate root:** `MeasurementDefinition`

Owns reference metadata and revision lineage for constructs, criteria, rubrics,
tasks, modality profiles, model revisions, validation studies, rights, and
provenance.

It stores references and contextual metadata, not raw responses, criterion
observations, rater parameters, score snapshots, or decisions.

## Context relationships

```text
Measurement Context Registry
       | Open Host Service: immutable references
       v
Rater Observation
       | Published Language: governed-rater observation v1
       v
Measurement Calibration
       | Customer/Supplier: parameter and score artifacts
       v
Assessment Operations
       | Domain Events: publication and adjudication outcomes
       v
Temporal Measurement Monitoring
```

The actual interaction is not a linear pipeline. Assessment Operations requests
observations and calibration, while every context resolves registry references.
The diagram shows authority flow only.

### Integration patterns

- **Published Language:** `fast-mlsirm` publishes the domain-neutral observation
  envelope. Consumers pin a released version and digest.
- **Anti-Corruption Layer:** every consumer translates its internal terms into
  the published language. Provider payloads, CEFR terminology, UI labels, and
  database rows never leak into the contract.
- **Open Host Service:** the registry exposes revision metadata without granting
  write access to its aggregate.
- **Customer/Supplier:** Assessment Operations requests numerical work but does
  not dictate estimator internals.
- **Separate Ways:** domain-specific profiles remain independent unless they can
  conform without weakening the core invariants.

A shared-kernel repository is intentionally not introduced. The cost of jointly
changing a shared kernel would exceed the benefit at this stage; released
published-language artifacts provide the required coupling instead.

## Aggregate invariants

### `RaterInvocation`

- exactly one reusable rater configuration;
- exactly one task and rubric revision;
- at least one criterion observation;
- no duplicate criterion observation;
- an observed category has one or more unique evidence references;
- an abstention has no manufactured category and has an explicit reason;
- failure and abstention remain in the denominator;
- no field for final score, latent trait, placement, pass/fail, certification, or
  employment decision.

### `RaterPanel`

- assignment identity is unique within a panel revision;
- publication freezes assignment and anchor membership;
- a configuration may have repeated invocations without becoming multiple
  independent raters;
- adjudication is a separate aggregate and never mutates source observations.

### `CalibrationDesign`

- target parameters and identification constraints are explicit;
- sampling design, error target, and failure denominator are versioned;
- model selection is evidence-based and not an unrecorded heuristic;
- every parameter snapshot points to the exact observation-panel digest.

### `RaterMonitoringRun`

- `available_at <= knowledge_cutoff` for every consumed artifact;
- configuration revisions remain distinguishable across time;
- invocation noise, gradual drift, and configuration change are separate
  estimands;
- monitoring results are immutable artifacts, not score rewrites.

## Domain events

| Event | Producer | Consumers |
|---|---|---|
| `rater_invocation_completed` | Rater Observation | Assessment Operations, Measurement Calibration |
| `rater_invocation_abstained` | Rater Observation | Assessment Operations, Measurement Calibration |
| `rater_invocation_failed` | Rater Observation | Assessment Operations, Measurement Calibration |
| `parameter_snapshot_published` | Measurement Calibration | Assessment Operations, Temporal Monitoring |
| `score_snapshot_published` | Assessment Operations | Temporal Monitoring, authorized product consumers |
| `adjudication_completed` | Assessment Operations | Temporal Monitoring, audit consumers |
| `rater_configuration_activated` | Measurement Registry | all contexts |
| `rater_configuration_retired` | Measurement Registry | all contexts |
| `rater_monitoring_artifact_published` | Temporal Monitoring | Assessment Operations, governance consumers |

Events carry opaque references, tenant and purpose context, occurred and
recorded times, correlation and causation identifiers, provenance, schema
version, and data classification. They do not broadcast response content or
PII.

## Migration from CEFR-specific work

CEFR is retained only as a future domain profile. Existing CEFR-specific branches
must not become the core model. Migration order:

1. publish this generic observation contract;
2. implement the generic observation bounded context;
3. implement panel operations and calibration contracts;
4. make any future CEFR work an Anti-Corruption Layer/profile over the released
   generic artifacts;
5. remove direct CEFR-to-numerical and CEFR-to-product coupling.

No CEFR descriptor text, level label, or overall-level decision belongs in this
core contract.

## Verification strategy

- Rust unit and property tests cover every aggregate rejection path;
- JSON Schema positive and negative fixtures prohibit decision leakage;
- true-parameter simulation validates severity, thresholds, discrimination,
  interactions, invocation variance, differential functioning, and uncertainty;
- CPU `f64` is the numerical reference and GPU results require parity evidence;
- downstream contract tests pin exact artifact digests;
- repeated model invocations are tested as nested runs rather than independent
  rater identities;
- failure, abstention, malformed output, and provider timeout remain explicit
  denominator states.

## References

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*. American Educational Research
Association.

Evans, E. (2003). *Domain-driven design: Tackling complexity in the heart of
software*. Addison-Wesley.

Myford, C. M., & Wolfe, E. W. (2003). Detecting and measuring rater effects
using many-facet Rasch measurement: Part I. *Journal of Applied Measurement,
4*(4), 386–422.

Myford, C. M., & Wolfe, E. W. (2004). Detecting and measuring rater effects
using many-facet Rasch measurement: Part II. *Journal of Applied Measurement,
5*(2), 189–227.

Vernon, V. (2013). *Implementing domain-driven design*. Addison-Wesley.
