# ADR-0100: Adaptive contextual-orchestrator mode is the LLM-judge default

Status: **Accepted**  
Date: 2026-08-17

## Context

`fast-mlsirm` delegates model transport and orchestration to an injected contextual-orchestrator adapter, while keeping rubric validation, result parsing, and psychometric projection inside this repository. The adapter previously defaulted ordinary judge calls to fixed `route` mode, which made the consumer choose execution topology even though contextual-orchestrator owns routing, verification, fallback, and cost policy.

The cross-repository boundary is already version-marked by `contextual-orchestrator-contract-v1`. Current live provider responses do not expose a separate immutable request/result artifact digest or mandatory provider/model identity fields, so this consumer must not invent those fields and reject otherwise valid live responses. A stronger result-schema version must first be published by the owning contextual-orchestrator contract and then adopted here fail-closed.

## Decision

`ContextualOrchestratorJudge` defaults to `mode="auto"`.

`auto` delegates execution topology to contextual-orchestrator. The orchestration plane may keep a simple task on one route or allocate deeper verification when its policy determines that additional reliability is warranted, while applying its own known-cost policy. Explicit `route` and `conduct` remain available for controlled ablation, incident response, and documented operational requirements.

Construction remains fail-closed on the public `contextual-orchestrator-contract-v1` marker. The caller continues to validate bounded model output and to preserve the returned validated orchestration mode, trace count, and usage evidence. `fast-mlsirm` does not infer provider/model provenance that the live upstream contract does not publish. When contextual-orchestrator publishes a stronger versioned request/result schema or immutable artifact digest, this ADR requires a compatibility update here before relying on that new evidence surface.

## Consequences

- Ordinary consumers no longer pin judge calls to fixed single-route execution.
- Explicit `route` and `conduct` behavior remains testable and observable.
- The adapter remains provider-neutral and keeps model output untrusted.
- Cross-repository compatibility remains fail-closed at the strongest contract actually published by the owning repository; no consumer-only response fields are fabricated.
- No psychometric numerical arithmetic moves out of Rust-owned production kernels.

## Evidence

Omidvar, H., & Akhlaghi, V. (2026). *A communication-theoretic framework for LLM agents: Cost-aware adaptive reliability* [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2605.09121

The study frames difficulty-adaptive routing as a reliability operator and reports a cost-aware semantic router that traverses an empirical quality-cost frontier, supporting per-task allocation rather than one fixed model-technique-budget choice.

Tang, Y., Cetin, E., Xu, J., Sun, Q., Nielsen, S., Richard, V., Goda, H., Tymchenko, I., Nguyen, N., Lee, H., Ashiga, M., Kotyan, S., Kuroki, S., & Clanuwat, T. (2026). *Sakana Fugu technical report* [Technical report]. arXiv. https://doi.org/10.48550/arXiv.2606.21228

The report describes query-adaptive agentic scaffolds and distinct latency-balanced versus quality-prioritized operating points, supporting adaptive orchestration instead of one fixed topology.
