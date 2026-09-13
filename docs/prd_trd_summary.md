# fast-mlsirm PRD/TRD Summary — Deprecated

Status: **Deprecated as an authoritative requirements source**  
Last reviewed: 2026-08-09

This file was the original narrow MVP summary for an MLS2PLM-focused prototype. The product has since expanded to governed assessment/rubric/scoring contracts, rubric-centered item generation, automated-scoring calibration/validation, enterprise adapters, broader diagnostics and release evidence. Several statements in the historical summary are therefore stale, including the earlier NumPy-primary/Rust-optional architecture and the earlier roadmap that treated ordinal/GPU work as unexplored future scope.

Use the following canonical documents instead:

- [Product Requirements Document](PRD.md)
- [Technical Requirements Document](TRD.md)
- [Root architecture description](../ARCHITECTURE.md)
- [Architecture Decision Record index](adr/README.md)
- [UML diagram index](uml/README.md)
- [Logical reusable-domain ERD](erd/domain-model.puml)
- [Requirements-to-implementation traceability](traceability/requirements-matrix.md)
- [Research-to-architecture traceability](traceability/research-basis.md)

## Historical scope retained for context

The initial product goal was fast simulation, fitting and recovery diagnostics for the simple-structure MLS2PLM specialization:

> The following equation is historical MVP notation only. It is retained to
> explain the origin of this deprecated summary and is not the current runtime
> contract. Current simple-structure MLS2PLM usage follows the canonical
> parameterization in [`docs/papers/mls2plm-canonical-equations.md`](papers/mls2plm-canonical-equations.md),
> including `a_i = exp(alpha_i)`, `gamma = exp(tau)`, the declared distance
> term, and the package's finite-value/epsilon rules.

```text
logit P(Y_pi = 1) = a_i * theta_p,d(i) + b_i - gamma * distance(xi_p, zeta_i)
```

The early architecture separated a Python API from a Rust numerical core and PyO3 binding. That separation remains conceptually valid, but the current governing architecture is stricter: production psychometric arithmetic is Rust-first, while Python owns validation/orchestration/reporting and governed reference/parity paths. Automatic backend resolution fails closed without the compiled Rust core.

The old MVP roadmap is intentionally not reproduced here because it is no longer the authoritative backlog. Current bounded requirements and proposed work are recorded in `PRD.md`, `TRD.md`, ADR statuses, open issues/PRs, and the traceability matrix.
