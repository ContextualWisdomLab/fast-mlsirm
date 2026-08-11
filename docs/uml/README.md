# UML and architecture diagrams

These PlantUML files are source-controlled architecture views. They are explanatory contracts, not generated screenshots.

| Diagram | Purpose |
|---|---|
| [`component.puml`](component.puml) | Repository/component ownership and external integration boundary. |
| [`scoring-sequence.puml`](scoring-sequence.puml) | Assessment/rubric -> scoring -> observation -> Rust calibration -> validation/report flow. |
| [`model-selection-sequence.puml`](model-selection-sequence.puml) | Relation-safe factor/model selection and recovery flow. |
| [`item-lifecycle.puml`](item-lifecycle.puml) | Governed item/rubric lifecycle and immutable revision semantics. |
| [`item-bank-state.puml`](item-bank-state.puml) | Compatibility alias for the canonical governed item lifecycle view. |
| [`deployment.puml`](deployment.puml) | Package deployment and downstream-host/service composition boundary. |
| [`domain-public-contract.puml`](domain-public-contract.puml) | Persistence-neutral domain/public-contract classes, construction invariants and host-adapter boundary. |

## Update rule

A change to ownership, public contract flow, model-selection decision logic, artifact lifecycle, or deployment/integration boundary must update the affected diagram in the same PR or document why the view is unchanged.

PlantUML sources are intentionally kept in the repository so diffs are reviewable and renderers can reproduce diagrams without sharing binary design assets.
