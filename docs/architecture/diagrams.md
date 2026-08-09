# Architecture diagrams

These Mermaid diagrams are normative views of the architecture described in `ARCHITECTURE.md`. They intentionally distinguish **implemented core**, **accepted/proposed reusable work**, and **downstream product ownership**. The ERD is logical: it models immutable artifacts and provenance, not a hosted ORM/database schema.

## 1. C4-style system context

```mermaid
flowchart LR
    Scientist[Psychometrician / researcher]
    Engineer[Assessment / AI-evaluation engineer]
    Commons[Psychometrics Commons\nHosted product]
    Core[fast-mlsirm\nReusable measurement core]
    Orchestrator[contextual-orchestrator\nOptional LLM orchestration]
    TEPP[TEPP / Gyeot\nTemporal and collection integrations]
    Portal[semantic-data-portal\nResearch release]

    Scientist --> Core
    Engineer --> Core
    Commons --> Core
    Core -. versioned provider protocol .-> Orchestrator
    Core -. versioned temporal handoff .-> TEPP
    Commons -. approved research release .-> Portal
```

`fast-mlsirm` does not own Commons' HTTP/session/consent/product database.

## 2. Component view

```mermaid
flowchart TB
    subgraph Python[Python package]
      PublicAPI[Public API / CLI]
      Contracts[Assessment / Rubric / Scoring contracts]
      Rubric[Rubric blueprint + generation validation]
      Adapters[Essay / RAG / enterprise adapters]
      Reports[Deterministic reports]
      Ref[NumPy reference/fallback paths]
    end

    subgraph Binding[Binding boundary]
      PyO3[Canonical PyO3 registry]
    end

    subgraph Rust[Rust numerical authority]
      Core[mlsirm-core]
      CPU[CPU f64 kernels]
      GPU[wgpu device kernels]
    end

    PublicAPI --> Contracts
    PublicAPI --> Rubric
    PublicAPI --> Adapters
    Adapters --> Contracts
    PublicAPI --> Reports
    PublicAPI --> PyO3
    Contracts --> PyO3
    PyO3 --> Core
    Core --> CPU
    Core --> GPU
    Ref -. parity / equation oracle .-> Core
```

## 3. Rubric-to-governed-item sequence

```mermaid
sequenceDiagram
    participant Author as Assessment author
    participant Rubric as Rubric compiler
    participant Provider as Untrusted provider
    participant Parser as Candidate validator
    participant Screen as Semantic screening
    participant Pilot as Human/artificial crowd
    participant Rust as Rust calibration
    participant Bank as Governed item bank

    Author->>Rubric: RubricSpecification + BlueprintPlan
    Rubric-->>Author: ItemBlueprint + GenerationContract
    Author->>Provider: bounded GenerationRequest
    Provider-->>Parser: untrusted JSON text
    Parser->>Parser: replay, schema, answer-key, source checks
    alt invalid structure/provenance
        Parser-->>Author: bounded failure code
    else valid candidate
        Parser-->>Screen: GeneratedItemCandidate
        Screen->>Screen: answerability, alignment, ambiguity, leakage, fairness
        alt screening failure
            Screen-->>Author: quarantine/reject evidence
        else screened
            Screen->>Pilot: pilot item
            Pilot-->>Rust: connected response/rater observations
            Rust-->>Bank: calibration, fit, DIF, information, uncertainty
            Bank->>Bank: approve / quarantine / retire decision
        end
    end
```

The current protected package implements the rubric/compiler/provider-validation boundary and many downstream calibration primitives. A complete hosted bank workflow is an accepted product direction, not a claim that this repository owns hosted persistence.

## 4. Automated scoring / fallible-rater sequence

```mermaid
sequenceDiagram
    participant Spec as AssessmentSpec
    participant Engines as Human / LLM / external scorers
    participant Obs as ScoreObservation contracts
    participant Facets as Rust many-facet calibration
    participant Valid as Validation / fairness
    participant Report as Deterministic report

    Spec->>Engines: exact rubric + engine policy
    Engines-->>Obs: scored / abstained / failed / excluded evidence
    Obs->>Facets: connected task-rater-criterion records
    Facets-->>Valid: latent calibration + rater evidence
    Valid->>Valid: agreement, DIF, subgroup/range/drift evidence
    Valid-->>Report: evidence + interpretation boundaries
```

## 5. Measurement-model selection flow

```mermaid
flowchart TD
    Purpose[Define intended score/use] --> Dim[Primary dimensionality candidates]
    Dim --> Base[Correlated MIRT / exploratory structure]
    Base --> LD{Residual local dependence?}
    LD -- shared stimulus/context --> Testlet[Testlet / two-tier candidate]
    LD -- no --> General{General factor claim?}
    Testlet --> General
    General -- yes --> BF[Bifactor candidate]
    General -- hierarchical --> HO[Higher-order candidate]
    General -- no --> Facet{Rater/task/occasion effects?}
    BF --> Facet
    HO --> Facet
    Facet -- yes --> MF[Many-facet extension]
    Facet -- no --> Relation[Classify model relation]
    MF --> Relation
    Relation --> Compare[LR / boundary bootstrap / formal Vuong path]
    Compare --> CV[Cluster-safe held-out prediction]
    CV --> Residual[Residual, DIF/invariance, stability]
    Residual --> Recovery[True-structure/parameter recovery]
    Recovery --> Scoreability[Scoreability / interpretability]
    Scoreability --> Decision{Evidence supports complexity?}
    Decision -- yes --> Accept[Versioned accepted model artifact]
    Decision -- no / indeterminate --> Simpler[Prefer simpler model or collect evidence]
```

A latent-space residual interaction is added only after substantive dimensions/facets/testlets when residual evidence justifies it.

## 6. Governed item-bank state machine

```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> Audited: author/content audit
    Audited --> Screened: structural + semantic gates
    Screened --> Pilot: approved for pilot
    Pilot --> Calibrated: connected responses + Rust fit
    Calibrated --> Approved: fit/DIF/info/recovery evidence
    Approved --> Active: release/serving decision
    Active --> Suspended: drift/DIF/security/quality concern
    Suspended --> Active: revalidated without identity rewrite
    Active --> Retired: end of intended use
    Suspended --> Retired: superseded or invalid
    Calibrated --> Quarantined: failed validity/fit evidence
    Screened --> Quarantined: failed semantic/fairness evidence
    Quarantined --> Draft: new revision only
    Retired --> [*]
```

Accepted item content is immutable. Repair creates a new revision/fingerprint rather than rewriting history.

## 7. Logical artifact ERD

```mermaid
erDiagram
    ASSESSMENT_SPEC ||--|{ CONSTRUCT_SPEC : declares
    ASSESSMENT_SPEC ||--|{ RUBRIC_VERSION : pins
    RUBRIC_SPECIFICATION ||--|{ RUBRIC_VERSION : versions
    RUBRIC_VERSION ||--|{ RUBRIC_CRITERION : contains
    RUBRIC_VERSION ||--o{ ITEM_BLUEPRINT : compiles
    ITEM_BLUEPRINT ||--o{ GENERATION_CONTRACT : binds
    GENERATION_CONTRACT ||--o{ GENERATION_REQUEST : instantiates
    GENERATION_REQUEST ||--o{ SOURCE_EVIDENCE : references
    GENERATION_REQUEST ||--o{ GENERATED_ITEM_CANDIDATE : produces
    GENERATED_ITEM_CANDIDATE ||--o{ CRITERION_OBSERVATION : administered_as
    ASSESSMENT_SPEC ||--o{ CRITERION_OBSERVATION : governs
    RUBRIC_CRITERION ||--o{ CRITERION_OBSERVATION : scores
    RATER_PROFILE ||--o{ CRITERION_OBSERVATION : emits
    CRITERION_OBSERVATION }o--|| CALIBRATION_ARTIFACT : informs
    CALIBRATION_ARTIFACT ||--o{ MODEL_COMPARISON_ARTIFACT : compared_by
    CALIBRATION_ARTIFACT ||--o{ RECOVERY_ARTIFACT : validated_by
    CALIBRATION_ARTIFACT ||--o{ SCOREABILITY_ARTIFACT : interpreted_by
    GENERATED_ITEM_CANDIDATE ||--o{ ITEM_BANK_REVISION : accepted_into
    CALIBRATION_ARTIFACT ||--o{ ITEM_BANK_REVISION : calibrates
    ITEM_BANK_REVISION ||--o{ RELEASE_EVIDENCE : released_with
    MODEL_COMPARISON_ARTIFACT ||--o{ RELEASE_EVIDENCE : supports
    RECOVERY_ARTIFACT ||--o{ RELEASE_EVIDENCE : supports
```

Entity names describe logical canonical artifacts only. A downstream service may map them to differently named persistence objects as long as exact identities and relationships are preserved.

## 8. Multilevel and temporal handoff

```mermaid
flowchart LR
    Obs[Observation] --> Member[Context membership design]
    Obs --> Occasion[Temporal occasion]
    Member --> Nested[Nested / cross-classified / multiple membership]
    Occasion --> Discrete[Discrete occasion-step dynamics]
    Occasion --> Continuous[Continuous-time / elapsed-gap model\nseparate parameterization]
    Nested --> Fit[Released Rust fitter only after recovery evidence]
    Discrete --> Fit
    Continuous --> Fit
    Fit --> Result[Context-aware measurement result]
```

The diagram records the required modeling boundary. It does not mark an unmerged dedicated multilevel/longitudinal namespace as released.

## 9. Release and provenance flow

```mermaid
flowchart LR
    Head[Exact protected head] --> CI[CI + security + owned coverage]
    CI --> Package[Rust/PyO3/wheel/package acceptance]
    Package --> Sci[Required recovery / scientific studies]
    Sci --> Review[Independent review + zero actionable findings]
    Review --> Prov[SBOM / provenance / immutable artifact identity]
    Prov --> Release[Version + CHANGELOG + release artifact]
    Release --> Verify[Install/execute released artifact verification]
```

Skipped, pending, predecessor-head, stale-base, synthetic-only, or missing evidence does not satisfy a release edge.
