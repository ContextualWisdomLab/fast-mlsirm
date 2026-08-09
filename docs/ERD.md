# fast-mlsirm Logical ERD / Domain Data Model

**Status:** Authoritative logical data model for reusable contracts.  
**Important:** `fast-mlsirm` does **not** own a hosted product database. This ERD documents domain relationships that downstream persistence may represent. It is not an instruction to add ORM models or migrations to this repository.

Database/persistence names shown here use descriptive two-or-more-word `snake_case` names. Durable public identities should be opaque/non-numeric where practical.

## 1. Logical ERD

```mermaid
erDiagram
  ASSESSMENT_SPEC ||--o{ ASSESSMENT_CONSTRUCT : defines
  ASSESSMENT_SPEC }o--|| RUBRIC_VERSION : uses
  RUBRIC_DEFINITION ||--o{ RUBRIC_VERSION : versions
  RUBRIC_VERSION ||--o{ RUBRIC_CRITERION : contains
  RUBRIC_CRITERION ||--o{ CRITERION_EVIDENCE : grounded_by
  RUBRIC_VERSION ||--o{ ITEM_BLUEPRINT : compiles
  ITEM_BLUEPRINT ||--o{ GENERATION_CONTRACT : binds
  GENERATION_CONTRACT ||--o{ GENERATED_ITEM_CANDIDATE : constrains
  GENERATED_ITEM_CANDIDATE ||--o{ ITEM_EVIDENCE : cites
  GENERATED_ITEM_CANDIDATE ||--o{ CRITERION_OBSERVATION : receives
  RUBRIC_CRITERION ||--o{ CRITERION_OBSERVATION : scores
  RATER_PROFILE ||--o{ CRITERION_OBSERVATION : produces
  EVALUATION_OCCASION ||--o{ CRITERION_OBSERVATION : occurs_at
  TARGET_RUN ||--o{ CRITERION_OBSERVATION : evaluated_as
  QUERY_TESTLET ||--o{ CRITERION_OBSERVATION : groups
  CRITERION_OBSERVATION ||--o{ CONTEXT_MEMBERSHIP : contextualized_by
  CONTEXT_DIMENSION ||--o{ CONTEXT_MEMBERSHIP : types
  CRITERION_OBSERVATION ||--o{ CALIBRATION_INPUT_LINK : contributes
  CALIBRATION_RUN ||--o{ CALIBRATION_INPUT_LINK : consumes
  CALIBRATION_RUN ||--o{ CALIBRATION_RESULT : produces
  CALIBRATION_RUN ||--o{ MODEL_COMPARISON_EVIDENCE : evaluated_by
  MODEL_SPECIFICATION ||--o{ CALIBRATION_RUN : fits
  ITEM_BANK_VERSION ||--o{ ITEM_BANK_ENTRY : contains
  GENERATED_ITEM_CANDIDATE ||--o{ ITEM_BANK_ENTRY : versioned_as
  CALIBRATION_RESULT ||--o{ ITEM_BANK_ENTRY : qualifies
  ITEM_BANK_VERSION ||--o{ BANK_ANCHOR_LINK : links
  RUBRIC_VERSION ||--o{ BANK_ANCHOR_LINK : anchors

  ASSESSMENT_SPEC {
    string assessment_spec_id PK
    string assessment_version
    string rubric_version_id FK
    string contract_fingerprint
  }
  ASSESSMENT_CONSTRUCT {
    string assessment_construct_id PK
    string assessment_spec_id FK
    string construct_name
    string construct_version
  }
  RUBRIC_DEFINITION {
    string rubric_definition_id PK
    string rubric_name
    string construct_namespace
  }
  RUBRIC_VERSION {
    string rubric_version_id PK
    string rubric_definition_id FK
    string semantic_version
    string rubric_fingerprint
    string lifecycle_status
  }
  RUBRIC_CRITERION {
    string rubric_criterion_id PK
    string rubric_version_id FK
    string construct_id
    string response_type
    string criterion_scope
    string criterion_criticality
    string evidence_scope
  }
  CRITERION_EVIDENCE {
    string criterion_evidence_id PK
    string rubric_criterion_id FK
    string evidence_reference
    string content_digest
  }
  ITEM_BLUEPRINT {
    string item_blueprint_id PK
    string rubric_version_id FK
    string blueprint_fingerprint
    string task_family
    string evidence_mode
    string difficulty_target
  }
  GENERATION_CONTRACT {
    string generation_contract_id PK
    string item_blueprint_id FK
    string contract_fingerprint
    string schema_version
  }
  GENERATED_ITEM_CANDIDATE {
    string item_candidate_id PK
    string generation_contract_id FK
    string candidate_fingerprint
    string candidate_status
  }
  ITEM_EVIDENCE {
    string item_evidence_id PK
    string item_candidate_id FK
    string source_reference
    string span_reference
    string source_digest
  }
  RATER_PROFILE {
    string rater_profile_id PK
    string rater_type
    string model_family
    string model_version
    string prompt_version
  }
  EVALUATION_OCCASION {
    string evaluation_occasion_id PK
    string occasion_order
    string timestamp_provenance
    string execution_fingerprint
  }
  TARGET_RUN {
    string target_run_id PK
    string target_system_id
    string run_seed_or_version
    string run_fingerprint
  }
  QUERY_TESTLET {
    string query_testlet_id PK
    string query_identity
    string evidence_regime
  }
  CRITERION_OBSERVATION {
    string criterion_observation_id PK
    string rubric_criterion_id FK
    string item_candidate_id FK
    string rater_profile_id FK
    string evaluation_occasion_id FK
    string target_run_id FK
    string query_testlet_id FK
    string observation_status
    string observed_value
  }
  CONTEXT_DIMENSION {
    string context_dimension_id PK
    string dimension_name
  }
  CONTEXT_MEMBERSHIP {
    string context_membership_id PK
    string criterion_observation_id FK
    string context_dimension_id FK
    string context_id
    float membership_weight
  }
  MODEL_SPECIFICATION {
    string model_specification_id PK
    string model_family
    string model_version
    string identification_contract
  }
  CALIBRATION_RUN {
    string calibration_run_id PK
    string model_specification_id FK
    string input_fingerprint
    string software_version
    string backend_device
  }
  CALIBRATION_INPUT_LINK {
    string calibration_input_link_id PK
    string calibration_run_id FK
    string criterion_observation_id FK
  }
  CALIBRATION_RESULT {
    string calibration_result_id PK
    string calibration_run_id FK
    string result_type
    string result_fingerprint
  }
  MODEL_COMPARISON_EVIDENCE {
    string model_comparison_evidence_id PK
    string calibration_run_id FK
    string relation_class
    string comparison_method
    string evidence_status
  }
  ITEM_BANK_VERSION {
    string item_bank_version_id PK
    string bank_name
    string bank_version
    string lifecycle_status
    string bank_fingerprint
  }
  ITEM_BANK_ENTRY {
    string item_bank_entry_id PK
    string item_bank_version_id FK
    string item_candidate_id FK
    string calibration_result_id FK
    string entry_status
  }
  BANK_ANCHOR_LINK {
    string bank_anchor_link_id PK
    string item_bank_version_id FK
    string rubric_version_id FK
    string anchor_identity
  }
```

## 2. Relationship semantics

### Assessment and rubric

An `assessment_spec` references one governed rubric revision for a given execution contract. Multiple assessment revisions can reuse the same rubric revision if the semantic scoring contract remains valid. A rubric definition has immutable versions rather than in-place edits.

### Rubric and item generation

A rubric version can compile many `item_blueprint` objects. A blueprint can produce one or more versioned `generation_contract` artifacts as the generation schema/provider constraints evolve. A candidate is accepted only against one exact generation contract and preserves its provenance.

### Evidence

`criterion_evidence` is evidence that defines or grounds a criterion. `item_evidence` is evidence used by a generated candidate. These are deliberately separate because a generated item can cite source spans that are not themselves the definition of the rubric criterion.

### Observations

`criterion_observation` is the normalized measurable event. It links the criterion/item, target/system-run, rater, occasion, and query/testlet. It may also link zero or more `context_membership` records across multiple context dimensions.

### Context and multiple membership

A context ID is not globally meaningful without `context_dimension_id`. A single observation may carry multiple memberships in one context dimension if the model/design permits weighted multiple membership. Weight validation is a contract concern, not a database trigger assumption.

### Calibration and model comparison

A `calibration_run` binds one exact model specification, input fingerprint, software version, and backend/device. Results are immutable evidence. Model comparison references fitted candidate evidence and records relation class/method/status rather than flattening all comparisons into one generic p-value.

### Item bank

An item bank is versioned. Entries refer to candidate identity and the calibration evidence used to admit them. Bank revisions use anchor/linking evidence rather than mutating the historical bank in place.

## 3. Automated scoring specialization

A human essay rater and an LLM essay scorer both map to `rater_profile`; the distinction is `rater_type` plus reproducible version metadata. Multiple rubric criteria share the same response target but remain distinct observations. A deterministic validation report is derived evidence; it is not the authoritative store of raw responses.

## 4. Reference-free RAG specialization

- RAG system configuration × run seed maps to `target_run`.
- Query and shared retrieved/evidence context maps to `query_testlet` and explicit evidence regime.
- Atomic groundedness/relevance/citation/robustness criteria map to `rubric_criterion`.
- Human/LLM judge model and prompt maps to `rater_profile` + `evaluation_occasion`.

The schema deliberately does not include a mandatory `reference_answer` because reference-free evaluation may not have one; evidence regime determines which claims are identifiable.

## 5. Enterprise issue specialization

Issue evidence and criteria can map to the same observation/rater/facet infrastructure. Stakeholder/business utility, intervention cost, uplift, and final resource-allocation decisions are **not** represented as item discrimination or calibration parameters. A downstream decision layer may reference calibration results.

## 6. Persistence boundary

A downstream service is free to normalize, denormalize, or event-source these relationships provided it preserves the public contract semantics. `fast-mlsirm` itself must not introduce a hidden product database merely to implement this ERD.

## 7. Naming and migration rules

- Persistent object names use two-or-more-word `snake_case` by default.
- Public IDs use opaque strings rather than auto-incremented numeric IDs where external exposure or cross-system correlation matters.
- Published artifacts are immutable; corrections create new revisions and supersession links.
- Breaking schema changes require explicit schema/version migration contracts.
