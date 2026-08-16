# ADR-0100: Adaptive contextual-orchestrator mode is the LLM-judge default

- Status: Accepted
- Date: 2026-08-16

## Context

The judge adapter delegated transport and orchestration to `contextual-orchestrator` but defaulted every call to fixed single-worker `route` mode. That prevented the orchestration plane from spending additional test-time compute when a difficult or high-risk judgment required verification, and it made the consumer—not the orchestration authority—choose the execution topology.

## Decision

`ContextualOrchestratorJudge` defaults to `mode="auto"`.

`auto` means the external orchestration plane owns model/provider selection, workflow depth, verification, fallback, and known-price optimization. Quality sufficiency is the first constraint; cost is minimized among execution paths that satisfy that constraint. The caller may still request `route` or `conduct` explicitly for controlled ablation, incident response, or a documented domain-specific requirement.

The adapter continues to own bounded rubric validation, strict duplicate-free JSON parsing, acceptance derivation, criterion-level output, and IRT projection. It does not infer quality or cost from the number of trace steps, and an unpriced model is not treated as free.

## Consequences

Ordinary consumers no longer pin the judge to one model call. Depending on task complexity and orchestrator policy, a judgment may use a single route or a deeper verified workflow. Returned orchestration mode, trace usage, and model provenance remain evidence and must be retained by the caller.

## References

Omidvar, H., & Akhlaghi, V. (2026). *A communication-theoretic framework for LLM agents: Cost-aware adaptive reliability* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2605.09121

The study unifies difficulty-adaptive reliability allocation and reports a cost-aware semantic router that traverses an empirical quality-cost frontier, supporting per-task allocation rather than one fixed model-technique-budget choice.

Tang, Y., Cetin, E., Xu, J., Sun, Q., Nielsen, S., Richard, V., Goda, H., Tymchenko, I., Nguyen, N., Lee, H., Ashiga, M., Kotyan, S., Kuroki, S., & Clanuwat, T. (2026). *Sakana Fugu technical report* [Technical report]. arXiv. https://doi.org/10.48550/arXiv.2606.21228

The report demonstrates query-adaptive worker selection and generated multi-agent scaffolds, with separate latency-balanced and quality-prioritized operating points that make the latency-quality trade-off explicit.
