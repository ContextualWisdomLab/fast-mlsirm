# UML Views — fast-mlsirm

These diagrams are logical views of the public product architecture. They intentionally avoid implying a hosted database or product runtime inside this repository.

## 1. Package/component UML

```mermaid
classDiagram
    class AssessmentSpec {
      +assessment_id: str
      +schema_version: str
      +rubric_fingerprint: str
      +calibration_contract
      +fairness_contract
      +reporting_contract
    }

    class RubricSpecification {
      +rubric_id: str
      +rubric_version: str
      +fingerprint: sha256
      +levels
      +response_format
    }

    class ItemBlueprint {
      +blueprint_id: str
      +construct_id: str
      +difficulty_target
      +evidence_mode
      +task_family
      +fingerprint: sha256
    }

    class GenerationContract {
      +contract_id: str
      +rubric_fingerprint: sha256
      +blueprint_fingerprint: sha256
      +json_schema
    }

    class GeneratedItemCandidate {
      +candidate_id: str
      +contract_fingerprint: sha256
      +answer_key
      +evidence_refs
      +status
    }

    class ScoreObservation {
      +observation_id: str
      +subject_id: str
      +item_or_task_id: str
      +criterion_id: str
      +rater_id: str
      +occasion_id: str
      +score
      +evidence_refs
    }

    class CalibrationResult {
      +model_family
      +parameterization
      +parameter_estimates
      +fit_evidence
      +uncertainty
      +provenance
    }

    class ModelComparisonResult {
      +relation
      +distinguishability
      +predictive_evidence
      +preferred_model
      +warnings
    }

    class ScoreabilityResult {
      +general_factor_evidence
      +specific_factor_evidence
      +interpretation_boundaries
    }

    class ItemBankRecord {
      +item_id: str
      +item_version: str
      +lifecycle_state
      +calibration_fingerprint
      +linking_anchor
      +exposure_policy
    }

    AssessmentSpec --> RubricSpecification
    RubricSpecification --> ItemBlueprint
    ItemBlueprint --> GenerationContract
    GenerationContract --> GeneratedItemCandidate
    GeneratedItemCandidate --> ScoreObservation
    ScoreObservation --> CalibrationResult
    CalibrationResult --> ModelComparisonResult
    CalibrationResult --> ScoreabilityResult
    CalibrationResult --> ItemBankRecord
```

## 2. Numerical ownership UML

```mermaid
classDiagram
    class PythonFacade {
      +validate()
      +marshal()
      +orchestrate()
      +render_report()
    }

    class PyO3Registry {
      +fit()
      +diagnostics()
      +rotation()
      +bifactor_scoreability()
      +future_bindings()
    }

    class RustCore {
      +likelihood()
      +gradient()
      +hessian_or_information()
      +optimize()
      +score()
      +compare_models()
      +recover()
    }

    class CpuBackend {
      +bounded_parallel_work()
      +deterministic_reductions()
    }

    class GpuBackend {
      +batched_kernels()
      +parity_evidence()
    }

    PythonFacade --> PyO3Registry
    PyO3Registry --> RustCore
    RustCore --> CpuBackend
    RustCore --> GpuBackend
```

## 3. Rubric-to-item sequence

```mermaid
sequenceDiagram
    actor Author as Assessment author
    participant Core as fast-mlsirm contracts
    participant Gen as Provider adapter / contextual-orchestrator
    participant Screen as Screening layer
    participant Crowd as Human/AI pilot
    participant Rust as Rust calibration
    participant Bank as Governed item-bank host

    Author->>Core: submit versioned RubricSpecification
    Core->>Core: validate, canonicalize, fingerprint
    Core->>Core: compile bounded blueprints + generation contracts
    Core->>Gen: provider-neutral generation request
    Gen-->>Core: untrusted structured candidate
    Core->>Screen: structural/evidence validation
    Screen-->>Core: screened candidate + reasons
    Core->>Crowd: administer accepted pilot items
    Crowd-->>Core: ScoreObservation[] with rater/occasion provenance
    Core->>Rust: calibration tensors + model contract
    Rust-->>Core: parameters, fit, uncertainty, recovery-compatible evidence
    Core->>Bank: immutable calibrated item/version artifact
```

## 4. Automated essay scoring sequence

```mermaid
sequenceDiagram
    actor Candidate
    participant Host as Hosted assessment product
    participant Scorer as Human/AI scoring adapter
    participant Core as fast-mlsirm scoring contracts
    participant Rust as Many-facet Rust calibration
    participant Report as Validation report
    actor Reviewer as Human adjudicator

    Candidate->>Host: submit essay
    Host->>Scorer: essay + prompt + rubric version
    Scorer-->>Core: criterion-level ScoreObservation + evidence
    Core->>Rust: linked essay/prompt/criterion/rater observations
    Rust-->>Core: severity, thresholds, fit, uncertainty
    Core->>Report: agreement, fairness/DIF, drift, review triggers
    alt human review required
      Report->>Reviewer: evidence-bounded adjudication packet
      Reviewer-->>Host: adjudication outcome
    else no review trigger
      Report-->>Host: calibrated evidence report
    end
```

## 5. Reference-free RAG measurement sequence

```mermaid
sequenceDiagram
    participant RAG as RAG system/run
    participant Evidence as Evidence builder
    participant Judge as Human/LLM judge family
    participant Core as fast-mlsirm observation layer
    participant Rust as Measurement core

    RAG->>Evidence: query + retrieved evidence + response identity
    Evidence->>Core: claims / obligations / provenance
    Core->>Judge: candidate-blind criterion requests
    Judge-->>Core: criterion observations + rater/prompt provenance
    Core->>Rust: multidimensional/testlet/facet observation tensor
    Rust-->>Core: calibrated latent quality + evaluator parameters
    Core->>Core: DIF, residual dependence, model comparison, scoreability
```

## 6. Multilevel and longitudinal interaction

```mermaid
sequenceDiagram
    participant Host
    participant Contract as Context/occasion contracts
    participant Rust as Rust estimator
    participant Recovery as Recovery suite

    Host->>Contract: observations + context dimensions + membership weights + occasions
    Contract->>Contract: exact-type, connectedness, provenance and temporal validation
    Contract->>Rust: sparse contextual/longitudinal design
    Rust->>Rust: likelihood / gradient / optimization
    Rust-->>Host: estimates + fit + uncertainty
    Recovery->>Rust: simulated true parameters and realistic designs
    Rust-->>Recovery: recovered parameters
    Recovery->>Recovery: align scale/rotation; bias, MAE, RMSE, coverage, convergence
```

## 7. Enterprise issue measurement and decision boundary

```mermaid
sequenceDiagram
    participant Source as Enterprise sources
    participant Extract as Evidence extraction
    participant Measure as Psychometric measurement
    participant Decide as Decision module / downstream product
    actor Human as Decision owner

    Source->>Extract: records/events/messages
    Extract-->>Measure: atomic issue + evidence/counterevidence + stakeholder observations
    Measure-->>Decide: latent state + uncertainty + disagreement + provenance
    Decide->>Decide: actions, outcome assumptions, costs, urgency, VOI
    Decide-->>Human: ranked action/review/information/monitoring queues
    Human-->>Decide: approved action / override / new evidence
```

## 8. Deployment/component boundary

```mermaid
flowchart LR
    subgraph fast[fast-mlsirm package]
      Py[Python API]
      Bind[PyO3]
      Rust[Rust core]
      Docs[Deterministic reports]
      Py --> Bind --> Rust
      Py --> Docs
    end

    subgraph host[Downstream hosted product]
      API[HTTP/Admin API]
      DB[(Persistence)]
      Auth[Identity/authorization]
      UI[Workbench/UI]
      API --> DB
      API --> Auth
      UI --> API
    end

    subgraph services[CWL services]
      Orch[contextual-orchestrator]
      Key[Keyverse]
      Eg[EgressWeave]
      Event[TEPP/Gyeot/etc.]
    end

    API --> Py
    API --> Orch
    API --> Key
    API --> Eg
    API --> Event
```

The hosted product imports versioned `fast-mlsirm` contracts; `fast-mlsirm` does not import hosted product code.
