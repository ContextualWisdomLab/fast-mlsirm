# Logical Contract ERD — fast-mlsirm

Status: architecture companion to `ARCHITECTURE.md`  
Important: this is a **logical information model**, not a physical database schema. `fast-mlsirm` does not require an ORM or own hosted-product persistence. A downstream product may persist these relationships under its own tenancy, authorization, migration, retention, encryption, and data-rights controls.

## 1. Core measurement/provenance ERD

```mermaid
erDiagram
    ASSESSMENT_SPEC ||--|{ CONSTRUCT_SPEC : declares
    CONSTRUCT_SPEC }o--o{ RUBRIC_SPECIFICATION : binds
    RUBRIC_SPECIFICATION ||--|{ RUBRIC_LEVEL : contains
    RUBRIC_SPECIFICATION ||--o{ ITEM_BLUEPRINT : compiles
    ITEM_BLUEPRINT ||--|| GENERATION_CONTRACT : produces
    GENERATION_CONTRACT ||--o{ GENERATED_ITEM_CANDIDATE : validates
    GENERATED_ITEM_CANDIDATE ||--o| ITEM_BANK_ENTRY : enters

    ASSESSMENT_SPEC ||--o{ SCORING_REQUEST : governs
    RUBRIC_SPECIFICATION ||--o{ SCORING_REQUEST : binds
    ENGINE_DESCRIPTOR ||--o{ SCORING_RESULT : executes
    SCORING_REQUEST ||--o| SCORING_RESULT : yields
    SCORING_RESULT ||--o{ SCORE_OBSERVATION : contains
    SCORE_OBSERVATION ||--o{ EVIDENCE_REFERENCE : cites

    ITEM_BANK_ENTRY ||--o{ PILOT_OBSERVATION : receives
    ENGINE_DESCRIPTOR ||--o{ PILOT_OBSERVATION : rates
    PILOT_OBSERVATION }o--|| CALIBRATION_RUN : informs
    CALIBRATION_RUN ||--o{ CALIBRATION_PARAMETER : estimates
    CALIBRATION_RUN ||--o{ MODEL_DIAGNOSTIC : emits
    CALIBRATION_RUN ||--o{ DIF_RESULT : emits
    CALIBRATION_RUN ||--o{ SCOREABILITY_RESULT : emits
    CALIBRATION_RUN ||--o{ MODEL_COMPARISON_RESULT : participates

    ASSESSMENT_SPEC {
      string assessment_id
      string assessment_version
      string assessment_fingerprint
      string assessment_handle
    }
    CONSTRUCT_SPEC {
      string construct_id
      string construct_definition_hash
    }
    RUBRIC_SPECIFICATION {
      string rubric_id
      string rubric_version
      string rubric_fingerprint
      string construct_id
    }
    RUBRIC_LEVEL {
      int score
      string level_label
      string descriptor_hash
    }
    ITEM_BLUEPRINT {
      string blueprint_id
      string blueprint_handle
      string blueprint_fingerprint
      string evidence_mode
      string difficulty_band
    }
    GENERATION_CONTRACT {
      string contract_id
      string contract_handle
      string contract_fingerprint
      string schema_version
    }
    GENERATED_ITEM_CANDIDATE {
      string candidate_id
      string candidate_fingerprint
      string contract_fingerprint
      string validation_status
    }
    ITEM_BANK_ENTRY {
      string item_entry_id
      string item_version
      string item_fingerprint
      string lifecycle_state
    }
    ENGINE_DESCRIPTOR {
      string engine_id
      string engine_family_id
      string engine_version
      string engine_kind
      string engine_fingerprint
    }
    SCORING_REQUEST {
      string request_id
      string request_fingerprint
      string task_revision_fingerprint
      string response_content_fingerprint
    }
    SCORING_RESULT {
      string result_id
      string result_fingerprint
      string request_fingerprint
      string engine_fingerprint
    }
    SCORE_OBSERVATION {
      string observation_id
      string observation_fingerprint
      string criterion_id
      string observation_status
      int score_category
    }
    EVIDENCE_REFERENCE {
      string source_id
      string span_id
      string content_fingerprint
      string evidence_role
    }
    PILOT_OBSERVATION {
      string pilot_observation_id
      string item_fingerprint
      string engine_fingerprint
      string respondent_or_system_id
      string occasion_id
    }
    CALIBRATION_RUN {
      string calibration_run_id
      string calibration_model_id
      string calibration_fingerprint
      string backend_identity
    }
    CALIBRATION_PARAMETER {
      string parameter_id
      string parameter_kind
      string target_identity
      string estimate_scale
    }
    MODEL_DIAGNOSTIC {
      string diagnostic_id
      string diagnostic_kind
      string evidence_status
    }
    DIF_RESULT {
      string dif_result_id
      string group_definition_id
      string item_fingerprint
    }
    SCOREABILITY_RESULT {
      string scoreability_result_id
      string model_family
      string interpretation_status
    }
    MODEL_COMPARISON_RESULT {
      string comparison_result_id
      string relation_class
      string distinguishability_status
      string preference_status
    }
```

## 2. Multilevel and temporal logical relationships

```mermaid
erDiagram
    OBSERVATION_UNIT ||--o{ CONTEXT_MEMBERSHIP : assigned_to
    CONTEXT_DIMENSION ||--o{ CONTEXT_MEMBERSHIP : scopes
    CONTEXT_ENTITY ||--o{ CONTEXT_MEMBERSHIP : receives
    OBSERVATION_UNIT ||--o{ TEMPORAL_OCCASION : observed_at
    TEMPORAL_OCCASION }o--|| LONGITUDINAL_DESIGN : ordered_by
    LONGITUDINAL_DESIGN ||--|| LONGITUDINAL_STATE_SPEC : uses

    OBSERVATION_UNIT {
      string observation_unit_id
      string revision_fingerprint
    }
    CONTEXT_DIMENSION {
      string context_dimension_id
      string context_dimension_version
    }
    CONTEXT_ENTITY {
      string context_dimension_id
      string context_id
      string context_revision_fingerprint
    }
    CONTEXT_MEMBERSHIP {
      string observation_unit_id
      string context_dimension_id
      string context_id
      float membership_weight
      string membership_fingerprint
    }
    TEMPORAL_OCCASION {
      string occasion_id
      int sequence_index
      string time_identity
      string revision_fingerprint
    }
    LONGITUDINAL_DESIGN {
      string longitudinal_design_id
      string design_fingerprint
    }
    LONGITUDINAL_STATE_SPEC {
      string state_spec_id
      string state_kind
      float autoregressive_coefficient
      bool random_intercept_enabled
      bool random_slope_enabled
    }
```

## 3. Persistence ownership rules

A downstream persistence implementation should preserve these semantics even if table names differ:

- durable public objects use opaque identifiers rather than sequential numeric public IDs;
- object/table names use descriptive two-or-more-word `snake_case` by default;
- full fingerprints are stored when replay/deduplication/audit depends on exact content identity;
- logical IDs and content fingerprints remain distinct;
- versioned content is append-only or superseded by a new version rather than mutated in place when historical interpretation matters;
- raw PII/source/response/prompt/provider text is not required by the core logical schema and should be stored only by an owning service under explicit purpose/retention controls;
- score/calibration/model evidence must remain linked to the exact rubric/task/item/engine/model revisions that produced it.

## 4. Suggested downstream physical names

These names are recommendations only for consumers that persist the contracts:

```text
assessment_specification
construct_definition
rubric_specification
rubric_level
item_blueprint
item_generation_contract
generated_item_candidate
item_bank_entry
scoring_request
scoring_result
score_observation
evidence_reference
engine_descriptor
pilot_observation
calibration_run
calibration_parameter
model_diagnostic
dif_result
scoreability_result
model_comparison_result
context_dimension
context_entity
context_membership
temporal_occasion
longitudinal_design
longitudinal_state_spec
```

Each name contains at least two descriptive tokens and follows the repository naming policy.

## 5. What is intentionally absent

This ERD does not define:

- user/account tables;
- tenant membership or RBAC tables;
- session/consent/result-access tables;
- billing/subscription tables;
- customer PII stores;
- LLM credential tables;
- deployment/runtime state;
- research publication catalog tables.

Those belong to the downstream product/service that owns the corresponding bounded context.
