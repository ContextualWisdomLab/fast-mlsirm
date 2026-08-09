# fast-mlsirm UML / Architecture Diagrams

**Status:** Authoritative diagram set  
**Notation:** Mermaid text diagrams kept in Git for review and drift detection.

Legend used below:

- `[MAIN]` — implemented on protected `main` at the time this baseline was cut.
- `[ACTIVE]` — represented by an open PR and not yet accepted on protected main.
- `[PLANNED]` — accepted architectural direction without complete protected-main
  implementation.
- `[DOWNSTREAM]` — belongs to another repository/bounded context.

## 1. Component diagram

```mermaid
flowchart TB
    subgraph Consumers[Consumers]
        PYC[Python client]
        CLI[CLI / research automation]
        HOST[psychometrics-commons\nDOWNSTREAM hosted product]
        RAG[RAG / LLM evaluation consumer]
        AES[Automated scoring consumer]
    end

    subgraph Fast[fast-mlsirm]
        CONTRACTS[MAIN\nAssessment / Rubric / Scoring contracts]
        RUBRIC[MAIN\nRubric blueprint + generation contracts]
        ADAPTERS[MAIN\nEssay / enterprise / provider-neutral adapters]
        ORCH[MAIN\nPython validation / orchestration / reports]
        BIND[MAIN\nPyO3 / numpy binding boundary]
        CORE[MAIN\nRust psychometric core]
        GPU[MAIN + expanding\nparity-gated GPU device kernels]
        REC[MAIN\nRecovery / fit / DIF / linking / evidence]
        BANK[PLANNED convergence\nGoverned item-bank lifecycle]
        MULTI[ACTIVE / evolving\nMultilevel + longitudinal contracts/models]
    end

    PYC --> CONTRACTS
    CLI --> ORCH
    HOST --> CONTRACTS
    RAG --> CONTRACTS
    AES --> CONTRACTS

    CONTRACTS --> RUBRIC
    CONTRACTS --> ADAPTERS
    RUBRIC --> ORCH
    ADAPTERS --> ORCH
    ORCH --> BIND
    BIND --> CORE
    CORE --> GPU
    CORE --> REC
    RUBRIC --> BANK
    BANK --> REC
    MULTI --> BIND

    KEY[Keyverse\nDOWNSTREAM identity] -.-> HOST
    TEPP[TEPP\nDOWNSTREAM event/trajectory] -.-> HOST
    CO[contextual-orchestrator\nDOWNSTREAM bounded AI] -.-> RAG
```

## 2. Numerical fitting sequence

```mermaid
sequenceDiagram
    participant U as Consumer
    participant P as Python API
    participant V as Validation/Contracts
    participant B as PyO3 Binding
    participant R as Rust Core
    participant G as GPU Device (optional)

    U->>P: fit(responses, factor_id, config)
    P->>V: validate shape/model/mask/resource contract
    V-->>P: normalized bounded inputs
    P->>B: call Rust numerical surface
    B->>R: typed arrays + validated configuration
    alt GPU requested and accepted for this kernel
        R->>G: dispatch parity-defined kernel
        G-->>R: device result/status
    else CPU
        R->>R: bounded fixed/coarse parallel compute
    end
    R-->>B: parameters, convergence, diagnostics/error
    B-->>P: typed result / stable error
    P-->>U: immutable/public result artifact
```

A Python reference/fallback may be retained for explicitly documented surfaces,
but it is not an independently evolving production formula.

## 3. Rubric-to-item lifecycle sequence

```mermaid
sequenceDiagram
    participant C as Construct / Assessment contract
    participant R as Rubric compiler
    participant G as Untrusted generator/provider
    participant S as Screening
    participant P as Pilot administration
    participant F as Rust calibration
    participant B as Governed item bank

    C->>R: RubricSpecification + evidence regime
    R->>R: compile bounded blueprint + generation contract
    R->>G: exact request + schema + provenance
    G-->>R: untrusted structured candidate
    R->>S: structurally rebound candidate
    S->>S: evidence + semantic + leakage/ambiguity checks
    S->>P: screened candidate set
    P->>F: human/artificial-crowd observations
    F-->>B: calibrated item evidence
    B->>B: approve/version/activate
    B-->>R: anchor/linking/drift feedback for next version
```

The generator never directly creates an `active` item. Structural JSON validity
is not psychometric validity.

## 4. Automated scoring / rater-calibration sequence

```mermaid
sequenceDiagram
    participant A as AssessmentSpec
    participant E as Human/AI Scoring Engine
    participant S as Shared Scoring Contract
    participant F as Facet / Measurement Calibration
    participant V as Validation / Fairness
    participant H as Human / Policy Review

    A->>S: ScoringRequest with exact rubric/task identity
    S->>E: bounded scoring input
    E-->>S: ScoreObservation(s)
    S->>S: replay engine/request/evidence provenance
    S->>F: scored + terminal observations
    Note over F: abstained/failed/excluded remain missing/terminal
    F-->>V: severity/criterion/calibration evidence
    V-->>H: agreement/range/DIF/drift/uncertainty triggers
    H-->>S: adjudication/policy outcome (downstream policy may own decision)
```

## 5. Measurement-model selection activity

```mermaid
flowchart TD
    CLAIM[Define intended score/use claim]
    DESIGN[Inspect design: dimensions/testlets/raters/context/time]
    CANDS[Build candidate structures]
    REL[Classify relation / boundary / overlap]
    TEST[Relation-appropriate inferential test]
    CV[Cluster-aware held-out prediction]
    LD[Residual local dependence + DIF/invariance]
    REC[True-structure + parameter recovery]
    SC[Scoreability / reliability / determinacy]
    OK{All required evidence defensible?}
    SIMPLE[Select simplest adequate model]
    NONE[Return indeterminate / revise design / collect evidence]

    CLAIM --> DESIGN --> CANDS --> REL --> TEST --> CV --> LD --> REC --> SC --> OK
    OK -- yes --> SIMPLE
    OK -- no --> NONE
```

## 6. Multilevel / multiple-membership / temporal handoff

```mermaid
sequenceDiagram
    participant O as Observation
    participant M as Membership contract
    participant T as Occasion contract
    participant R as Rust measurement model
    participant E as Evidence artifact
    participant TEPP as TEPP (DOWNSTREAM)

    O->>M: dimension-qualified contexts + exact weights
    O->>T: respondent + occasion + ordering/revision
    M->>M: connectedness/integrity/resource validation
    T->>T: ordering/state-spec integrity validation
    M->>R: reusable measurement design when estimator supported
    T->>R: repeated-measurement design when estimator supported
    R-->>E: calibrated measurement evidence
    E-->>TEPP: versioned score/occasion artifact for broader trajectory analysis
```

Time metadata alone does not authorize a continuous-time likelihood; the model
must explicitly implement and recover that parameterization.

## 7. Governed item-bank state diagram

```mermaid
stateDiagram-v2
    [*] --> Draft
    Draft --> Audited: structural/provenance audit
    Audited --> Screened: semantic/evidence screening
    Screened --> Pilot: approved for pilot administration
    Pilot --> Calibrated: sufficient calibration evidence
    Calibrated --> Approved: scientific/governance review
    Approved --> Active: immutable release published
    Active --> Suspended: drift/DIF/exposure/security concern
    Suspended --> Active: revalidated without changing release identity
    Suspended --> Retired: no longer defensible
    Active --> Retired: planned end of life
    Draft --> Retired: rejected candidate
    Audited --> Retired: failed audit
    Screened --> Retired: failed screen
    Pilot --> Retired: failed calibration
```

A changed criterion/evidence/scoring contract creates a **new version** rather
than mutating an active release back to an earlier lifecycle state.

## 8. Release-evidence sequence

```mermaid
sequenceDiagram
    participant H as Exact protected head
    participant CI as CI / Security / Fuzz
    participant SCI as Recovery / Statistical studies
    participant PKG as Package builder
    participant DD as Due-diligence / Evidence index
    participant REL as Release

    H->>CI: run required exact-head gates
    H->>SCI: run applicable recovery/study gates
    CI-->>PKG: success + provenance
    SCI-->>PKG: accepted scientific artifacts
    PKG->>PKG: build wheel/sdist and reinstall smoke
    PKG-->>DD: artifact hashes + acceptance evidence
    DD->>DD: SBOM/provenance/buyer/rollback checks
    DD-->>REL: one exact release evidence bundle
```

Queued, predecessor-head, cancelled or synthetic evidence cannot replace the
exact release-artifact chain.

## 9. Ecosystem boundary diagram

```mermaid
flowchart LR
    FAST[fast-mlsirm\nmeasurement contracts + Rust psychometrics]
    PC[psychometrics-commons\nhosted lifecycle/persistence/UI]
    KEY[Keyverse\nidentity/federation]
    TEPP[TEPP\ntemporal/event/trajectory]
    GYEOT[Gyeot\nEMA/ESM collection]
    SDP[semantic-data-portal\nresearch release/catalog]
    CO[contextual-orchestrator\nbounded LLM orchestration]
    EGRESS[EgressWeave\nexternal egress boundary]

    PC --> FAST
    PC --> KEY
    PC --> TEPP
    PC --> SDP
    GYEOT --> TEPP
    CO --> FAST
    CO --> EGRESS

    FAST -. forbidden reverse dependency .-> PC
```

The final dashed edge describes a prohibited dependency, not a runtime call.
