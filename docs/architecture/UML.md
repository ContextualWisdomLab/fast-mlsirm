# UML and Behavioral Views — fast-mlsirm

Status: architecture companion to `ARCHITECTURE.md`  
Notation: Mermaid diagrams aligned conceptually with UML 2.5.1; diagrams are documentation models, not code-generation schemas.

## 1. Component view

```mermaid
flowchart TB
    subgraph Public[Public surface]
      PAPI[Python API]
      CLI[CLI]
      REPORTS[JSON / HTML reports]
    end

    subgraph Contracts[Governed contracts]
      AS[AssessmentSpec]
      RS[RubricSpecification]
      SC[Scoring contracts]
      ML[Multilevel / temporal contracts]
    end

    subgraph Authoring[Item authoring]
      BP[Blueprint compiler]
      GC[Generation contract]
      CV[Candidate validation]
      SCREEN[Screening contracts]
      BANK[Item-bank lifecycle contracts]
    end

    subgraph Measurement[Measurement orchestration]
      FIT[Fit / calibration facade]
      DIAG[Fit / DIF / invariance / reliability]
      MODELSEL[Factor retention / model comparison]
      ROT[Rotation selection]
    end

    subgraph Native[Native implementation]
      PYO3[PyO3 bindings]
      RUST[mlsirm-core]
      GPU[wgpu device kernels]
    end

    PAPI --> Contracts
    PAPI --> Authoring
    PAPI --> Measurement
    CLI --> PAPI
    Contracts --> Authoring
    Contracts --> Measurement
    Authoring --> Measurement
    Measurement --> PYO3
    PYO3 --> RUST
    RUST --> GPU
    Measurement --> REPORTS
```

## 2. Contract class view

```mermaid
classDiagram
    class AssessmentSpec {
      +assessment_id
      +assessment_version
      +assessment_fingerprint
      +constructs
      +policy_refs
    }

    class ConstructSpec {
      +construct_id
      +construct_definition
      +rubric_fingerprints
    }

    class RubricSpecification {
      +rubric_id
      +rubric_version
      +construct_id
      +response_format
      +levels
      +fingerprint
    }

    class RubricLevel {
      +score
      +label
      +descriptor
      +observable_indicators
    }

    class ItemBlueprint {
      +blueprint_id
      +blueprint_handle
      +blueprint_fingerprint
      +generation_seed
      +difficulty_band
      +evidence_mode
    }

    class GenerationContract {
      +contract_id
      +contract_handle
      +contract_fingerprint
      +json_schema
    }

    class EngineDescriptor {
      +engine_id
      +engine_family_id
      +engine_kind
      +engine_version
      +model_id
      +prompt_template_fingerprint
    }

    class ScoringRequest {
      +request_id
      +assessment_fingerprint
      +rubric_fingerprint
      +task_revision_fingerprint
      +response_content_fingerprint
    }

    class ScoreObservation {
      +observation_id
      +criterion_id
      +status
      +score_category
      +observation_fingerprint
    }

    class ScoringResult {
      +result_id
      +request_fingerprint
      +engine_fingerprint
      +observations
      +result_fingerprint
    }

    class EvidenceReference {
      +source_id
      +span_id
      +content_fingerprint
      +evidence_role
    }

    AssessmentSpec "1" o-- "1..*" ConstructSpec
    ConstructSpec "1" --> "1..*" RubricSpecification : references fingerprints
    RubricSpecification "1" o-- "1..*" RubricLevel
    RubricSpecification "1" --> "0..*" ItemBlueprint : compiles
    ItemBlueprint "1" --> "1" GenerationContract
    AssessmentSpec "1" --> "0..*" ScoringRequest
    RubricSpecification "1" --> "0..*" ScoringRequest
    EngineDescriptor "1" --> "0..*" ScoringResult
    ScoringRequest "1" --> "0..1" ScoringResult
    ScoringResult "1" o-- "0..*" ScoreObservation
    ScoreObservation "1" o-- "0..*" EvidenceReference
```

## 3. Item-generation and calibration sequence

```mermaid
sequenceDiagram
    actor Owner as Assessment owner
    participant R as Rubric compiler
    participant P as Provider adapter
    participant V as Candidate validator
    participant S as Semantic screener
    participant A as Artificial crowd / human pilot
    participant C as Rust calibration
    participant B as Governed item bank

    Owner->>R: approve RubricSpecification + BlueprintPlan
    R-->>Owner: ItemBlueprint + GenerationContract fingerprints
    Owner->>P: generation contract + bounded source packet
    P-->>V: untrusted provider JSON
    V->>V: structural/provenance/evidence checks
    V-->>S: immutable candidate
    S->>S: construct/answerability/ambiguity/bias/leakage checks
    S-->>A: screened candidate
    A-->>C: bounded item/rater response observations
    C-->>B: difficulty/discrimination/fit/DIF/information + uncertainty
    B-->>Owner: approve / quarantine / reject / activate
```

## 4. Scoring execution sequence

```mermaid
sequenceDiagram
    participant App as Standalone/downstream app
    participant Spec as AssessmentSpec
    participant Eng as Human/Automated ScoringEngine
    participant Obs as Observation validator
    participant Rust as Rust calibration core
    participant Val as Validation/fairness
    participant Rep as Audit/report

    App->>Spec: load exact assessment + rubric fingerprints
    App->>Eng: ScoringRequest
    Eng-->>Obs: ScoreObservation(s)
    Obs->>Obs: replay + provenance + status validation
    Obs-->>Rust: typed criterion/rater observations
    Rust-->>Val: calibrated parameters / uncertainty
    Val-->>Rep: validity, fairness, drift, insufficient-evidence states
    Rep-->>App: JSON/HTML evidence bundle
```

## 5. Model-selection activity view

```mermaid
flowchart TD
    DATA[Observed responses + design metadata] --> RETAIN[Primary factor-retention evidence]
    RETAIN --> CANDS[Candidate structural models]
    CANDS --> REL{Classify actual relation}
    REL -->|regular nested| LR[LR / robust LR]
    REL -->|boundary / singular| BLR[Parametric-bootstrap / boundary-aware LR]
    REL -->|strictly non-nested / overlapping| DIST[Formal distinguishability]
    REL -->|unknown| STOP[No preference]
    DIST -->|distinguishable| VUONG[Vuong selection statistic]
    DIST -->|not distinguishable| SIMPLE[Prefer simpler / indeterminate]
    LR --> PRED[Held-out / clustered predictive evidence]
    BLR --> PRED
    VUONG --> PRED
    PRED --> REC[True-structure / parameter recovery]
    REC --> SCORE[Scoreability / invariance / DIF]
    SCORE --> DECIDE{Evidence materially better?}
    DECIDE -->|yes| SELECT[Select candidate]
    DECIDE -->|no / equivalent| SIMPLE
```

## 6. Item-bank state machine

```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> Audited: content/evidence audit passes
    Audited --> Screened: structural + semantic screening passes
    Screened --> Piloting: pilot assigned
    Piloting --> Calibrated: calibration evidence sufficient
    Calibrated --> Approved: governance approval
    Approved --> Active: published in bank/form
    Active --> Suspended: drift/DIF/anchor/security concern
    Suspended --> Active: revalidated
    Suspended --> Retired: unresolved defect
    Active --> Retired: planned retirement
    Draft --> Retired: rejected
    Audited --> Retired: rejected
    Screened --> Retired: rejected
    Piloting --> Retired: poor fit/information/fairness
```

## 7. Multilevel and temporal contract view

```mermaid
classDiagram
    class ContextMembership {
      +observation_id
      +context_dimension_id
      +context_id
      +membership_weight
      +fingerprint
    }
    class ContextMembershipDesign {
      +observation_ids
      +context_dimensions
      +context_keys
      +design_fingerprint
    }
    class TemporalOccasion {
      +occasion_id
      +sequence_index
      +timestamp_or_offset
      +revision_fingerprint
    }
    class LongitudinalStateSpec {
      +state_kind
      +random_intercept
      +random_slope
      +autoregressive_coefficient
    }
    class LongitudinalDesign {
      +respondent_id
      +occasions
      +state_spec
      +design_fingerprint
    }

    ContextMembershipDesign "1" o-- "1..*" ContextMembership
    LongitudinalDesign "1" o-- "1..*" TemporalOccasion
    LongitudinalDesign "1" o-- "1" LongitudinalStateSpec
```

The temporal model above is a contract view. A discrete occasion-step autoregressive coefficient must not be interpreted as a continuous-time process without a distinct likelihood/transition model.

## 8. Deployment view

```mermaid
flowchart LR
    subgraph Local[Standalone]
      U[User] --> PY[fast-mlsirm Python/CLI]
      PY --> NATIVE[PyO3 + Rust core]
    end

    subgraph Hosted[Downstream hosted product]
      C[Client] --> API[Owning product API]
      API --> WORKER[Measurement worker importing fast-mlsirm]
      API --> DB[(Owning product database)]
      API -. optional .-> ORCH[contextual-orchestrator]
      API -. optional .-> TEMP[TEPP]
      API -. optional .-> EGRESS[EgressWeave]
    end
```

No hosted topology in this diagram is a runtime requirement of `fast-mlsirm` itself.

## 9. Diagram maintenance rules

- Every diagram must distinguish implemented contracts from roadmap elements in the accompanying prose.
- UML/diagram element names should match public package terms when those terms already exist.
- Changes that alter bounded-context ownership, core data flow, numerical ownership, or contract identity require an ADR and updates to this file and `ARCHITECTURE.md`.
- Physical persistence belongs in downstream architecture; this repository keeps only the logical contract ERD in `docs/architecture/ERD.md`.

## Reference

Object Management Group. (2017). *OMG Unified Modeling Language (OMG UML), Version 2.5.1*.
