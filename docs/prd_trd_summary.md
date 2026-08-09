# fast-mlsirm PRD/TRD Summary

> **Compatibility note:** this path is retained because historical links point to it. The previous MLS2PLM-only MVP summary is no longer authoritative and had become materially stale relative to protected-main scoring, rubric, model-comparison, bifactor, rotation, reporting, and enterprise/essay validation capabilities.

The authoritative documentation set is now:

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — architecture, bounded contexts, scientific/model hierarchy, security and quality gates.
- [`PRD.md`](PRD.md) — current product requirements and buyer outcomes.
- [`TRD.md`](TRD.md) — technical contracts, numerical authority, validation, security, runtime, model-selection, and interoperability requirements.
- [`adr/README.md`](adr/README.md) — architecture decision records.
- [`UML.md`](UML.md) — component, class, sequence, state, and deployment views.
- [`ERD.md`](ERD.md) — persistence-agnostic logical domain/contract model.
- [`traceability.md`](traceability.md) — research/conversation requirement to code/status traceability.

## Current one-paragraph product definition

`fast-mlsirm` is an independently installable, Rust-first psychometric and measurement toolkit. It owns reusable Assessment/Rubric/Scoring contracts, observations, calibration and model diagnostics, linking/equating, DIF/invariance/fairness, factor/model selection, recovery/simulation, rater/judge calibration, automated-scoring validation, rubric/item-generation governance primitives, and related numerical kernels. It does **not** own the hosted Psychometrics Commons HTTP/session/consent/product-database/UI/deployment runtime.

## Current architecture shorthand

```text
Reusable assessment/rubric/scoring contracts
        ↓
Bounded Python validation/orchestration
        ↓
PyO3
        ↓
Rust numerical source of truth
        ↓
CPU parallel / parity-proven GPU execution
        ↓
Diagnostics, model selection, calibration and deterministic evidence
        ↓
Versioned downstream contracts / governed measurement artifacts
```

The retained NumPy code is a bounded reference/fallback and parity surface; it is not the architectural authority for new production numerical work.

## Update rule

Do not expand this compatibility summary into a second PRD/TRD. Material requirements belong in the authoritative files above so product scope does not fragment again.
