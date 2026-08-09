# fast-mlsirm UML and Interaction Views

**Status:** Authoritative diagram set accompanying `ARCHITECTURE.md`.  
**Notation:** Mermaid diagrams are maintained as text so review history and architecture changes remain diffable.

## 1. Component view

```mermaid
flowchart TB
  subgraph Public[Python public package]
    API[Public API / CLI]
    AS[Assessment & Scoring contracts]
    RU[Rubric / Item-generation contracts]
    EV[Evidence / Observation adapters]
    RP[Reports]
  end

  subgraph Compute[Scientific compute]
    PY[PyO3 binding registry]
    CORE[Rust mlsirm-core]
    FIT[Estimators]
    DIA[Diagnostics / DIF / linking]
    SEL[Factor retention / model comparison / rotation]
    REC[Simulation / recovery]
  end

  subgraph Device[Execution]
    CPU[CPU / parallel kernels]
    GPU[GPU kernels with parity evidence]
  end

  subgraph Optional[Explicit external integrations]
    LLM[LLM/provider adapter]
    ORCH[contextual-orchestrator]
  end

  API --> AS
  API --> RU
  AS --> EV
  RU --> EV
  EV --> PY
  PY --> CORE
  CORE --> FIT
  CORE --> DIA
  CORE --> SEL
  CORE --> REC
  FIT --> CPU
  DIA --> CPU
  SEL --> CPU
  REC --> CPU
  FIT -. when supported .-> GPU
  DIA -. when supported .-> GPU
  SEL -. when supported .-> GPU
  REC -. when supported .-> GPU
  EV --> RP
  RU -. optional .-> LLM
  LLM -. provider routing .-> ORCH
```

## 2. Package dependency view

```mermaid
flowchart LR
  INIT[fast_mlsirm.__init__] --> SC[scoring]
  INIT --> RUB[rubric]
  INIT --> EST[estimators]
  INIT --> DIAG[diagnostics]
  INIT --> LINK[linking/equating]
  INIT --> CAT[CAT/ATA]
  INIT --> REP[reporting]
  SC --> RUB
  SC --> BIND[PyO3 bindings]
  EST --> BIND
  DIAG --> BIND
  LINK --> BIND
  CAT --> BIND
  BIND --> RUST[mlsirm-core]
```

Dependency direction is intentional. Domain-specific scoring modules may reuse generic rubric/assessment types. The rubric package must not depend on hosted product state. Numerical code must not call back into downstream hosted services.

## 3. Core domain class view

```mermaid
classDiagram
  class AssessmentSpec {
    +assessment_id
    +assessment_version
    +constructs
    +rubric_ref
    +calibration_contract
    +validity_contract
  }

  class RubricSpecification {
    +rubric_id
    +rubric_version
    +fingerprint
    +levels
    +response_format
  }

  class RubricCriterion {
    +criterion_id
    +construct_id
    +response_type
    +criticality
    +evidence_scope
  }

  class ItemBlueprint {
    +blueprint_id
    +rubric_fingerprint
    +difficulty_target
    +task_family
    +evidence_mode
  }

  class GenerationContract {
    +contract_id
    +blueprint_id
    +rubric_fingerprint
    +output_schema
  }

  class CriterionObservation {
    +observation_id
    +criterion_id
    +target_id
    +rater_id
    +occasion_id
    +value
    +status
  }

  class RaterProfile {
    +rater_id
    +rater_type
    +model_version
    +prompt_version
  }

  class CalibrationRun {
    +calibration_run_id
    +model_family
    +model_version
    +input_fingerprint
  }

  class ItemBankVersion {
    +item_bank_version_id
    +status
    +anchor_set
    +calibration_ref
  }

  AssessmentSpec --> RubricSpecification : references
  RubricSpecification "1" --> "many" RubricCriterion : defines
  RubricSpecification --> ItemBlueprint : compiles
  ItemBlueprint --> GenerationContract : binds
  RubricCriterion --> CriterionObservation : scored by
  RaterProfile --> CriterionObservation : produces
  CriterionObservation --> CalibrationRun : calibrates
  CalibrationRun --> ItemBankVersion : supports
```

The class view is conceptual. Exact Python type names and fields are governed by source; the logical domain must not be mistaken for a SQL schema.

## 4. Rubric-to-item sequence

```mermaid
sequenceDiagram
  autonumber
  actor Author as Assessment author
  participant API as fast_mlsirm public API
  participant Comp as Rubric/Blueprint compiler
  participant Prov as Provider adapter
  participant Parse as Candidate validator
  participant Pilot as Pilot administrator
  participant Rust as Rust calibration core
  participant Bank as Governed item bank

  Author->>API: AssessmentSpec + RubricSpecification
  API->>Comp: validate/version/compile
  Comp-->>API: ItemBlueprint + GenerationContract
  API->>Prov: bounded provider request
  Prov-->>API: untrusted JSON
  API->>Parse: parse + provenance replay + evidence checks
  alt candidate rejected
    Parse-->>API: structured failure
  else candidate structurally accepted
    Parse-->>Pilot: candidate + immutable provenance
    Pilot->>Pilot: semantic/content screening
    Pilot->>Rust: observations + design metadata
    Rust-->>Pilot: item/rater/model calibration evidence
    Pilot->>Bank: accept/quarantine/retire proposal
  end
```

## 5. Automated-scoring sequence

```mermaid
sequenceDiagram
  autonumber
  participant T as Target response
  participant H as Human rater(s)
  participant A as Automated scorer(s)
  participant O as Observation contract
  participant M as Many-facet / structural model
  participant V as Validation engine
  participant R as Evidence report

  T->>H: scoring task + rubric version
  T->>A: same governed scoring task
  H-->>O: human judgments + rater identity
  A-->>O: automated judgments + model/prompt identity
  O->>M: linked person/item/rater/occasion observations
  M-->>V: calibrated estimates + diagnostics
  V->>V: agreement, error, DIF, range use, drift, invariance
  V-->>R: deterministic JSON/HTML evidence + interpretation boundaries
```

## 6. Reference-free RAG sequence

```mermaid
sequenceDiagram
  autonumber
  participant Q as Query
  participant E as Independent evidence regime
  participant S as RAG system/run
  participant C as Criterion builder
  participant J as Human/LLM judges
  participant P as Psychometric calibration

  Q->>E: establish evidence universe
  E->>C: obligations/nuggets/anchors
  C-->>J: candidate-blind criterion set
  Q->>S: execute retrieval/generation
  S-->>J: response + retrieved evidence
  J-->>P: criterion observations + judge/prompt/occasion IDs
  P-->>P: facets + multidimensional/testlet model
  P-->>P: latent-space residual only if supported
```

A target response used to discover a candidate-aware criterion cannot be scored by that same discovery criterion without an independent fold if the result is intended for comparative benchmarking.

## 7. Structural model-selection activity

```mermaid
stateDiagram-v2
  [*] --> DefineInterpretation
  DefineInterpretation --> PrimaryDimensionality
  PrimaryDimensionality --> DiagnoseLocalDependence
  DiagnoseLocalDependence --> AddTestletOrSecondary: shared stimulus/query dependence
  DiagnoseLocalDependence --> AddFacets: rater/task/occasion effects
  DiagnoseLocalDependence --> GeneralFactorCandidates: general-score claim
  AddTestletOrSecondary --> RelationClassification
  AddFacets --> RelationClassification
  GeneralFactorCandidates --> RelationClassification
  RelationClassification --> RegularLR: regular nested
  RelationClassification --> BootstrapLR: boundary/nonlinear nested
  RelationClassification --> Distinguishability: non-nested/overlapping
  RelationClassification --> Indeterminate: unknown
  Distinguishability --> VuongSelection: distinguishable
  Distinguishability --> PreferSimpler: observationally indistinguishable
  RegularLR --> PredictiveValidation
  BootstrapLR --> PredictiveValidation
  VuongSelection --> PredictiveValidation
  PreferSimpler --> PredictiveValidation
  PredictiveValidation --> ResidualAndScoreability
  ResidualAndScoreability --> Recovery
  Recovery --> SelectSimplestAdequate
  SelectSimplestAdequate --> [*]
  Indeterminate --> [*]
```

## 8. Governed item-bank lifecycle

```mermaid
stateDiagram-v2
  [*] --> Draft
  Draft --> Audited
  Audited --> Screened
  Screened --> Pilot
  Pilot --> Calibrated
  Calibrated --> Approved
  Approved --> Active
  Active --> Suspended: anomaly / drift / policy hold
  Active --> Retired: planned retirement
  Suspended --> Active: evidence clears hold
  Suspended --> Retired: failure confirmed
  Calibrated --> Quarantined: fit/DIF/evidence failure
  Quarantined --> Pilot: revised candidate
  Retired --> [*]
```

Operational artifacts are immutable. Revision creates a new version rather than mutating an active object.

## 9. Multilevel / multiple-membership view

```mermaid
classDiagram
  class Observation {
    +observation_id
    +target_id
    +item_id
    +rater_id
    +occasion_id
  }
  class ContextDimension {
    +context_dimension_id
    +name
  }
  class ContextMembership {
    +membership_id
    +context_dimension_id
    +context_id
    +weight
  }
  class TemporalOccasion {
    +occasion_id
    +order
    +timestamp_provenance
  }
  class Testlet {
    +testlet_id
  }

  Observation "1" --> "many" ContextMembership
  ContextDimension "1" --> "many" ContextMembership
  Observation --> TemporalOccasion
  Observation --> Testlet
```

Context weight semantics, discrete occasion semantics, and any continuous-time parameterization are separate contracts.

## 10. Deployment/boundary view

```mermaid
flowchart LR
  subgraph Consumer[Downstream product/service]
    PC[Psychometrics Commons or another consumer]
    DB[(Product-owned persistence)]
    UI[Product UI/API]
    UI --> DB
  end

  subgraph Library[fast-mlsirm package]
    PY[Python API]
    EXT[PyO3 extension]
    CORE[Rust core]
    PY --> EXT --> CORE
  end

  subgraph Adjacent[Adjacent bounded services]
    ID[Keyverse]
    OR[contextual-orchestrator]
    EG[EgressWeave]
    SD[semantic-data-portal]
  end

  PC --> PY
  PC --> DB
  PC -. identity .-> ID
  PY -. optional LLM orchestration .-> OR
  OR -. controlled egress .-> EG
  PC -. research/release artifacts .-> SD
```

There is no shared product database owned by `fast-mlsirm`.

## 11. Diagram maintenance rule

Update these diagrams when a material PR changes:

- bounded-context ownership;
- package dependency direction;
- public contract lifecycle;
- model-selection flow;
- context/temporal semantics;
- execution authority or device ownership;
- hosted/downstream integration boundaries.
