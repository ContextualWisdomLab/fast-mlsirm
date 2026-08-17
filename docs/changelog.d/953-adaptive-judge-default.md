# Default LLM judge orchestration to adaptive auto mode

## Changed

- `ContextualOrchestratorJudge` now defaults ordinary calls to contextual-orchestrator `auto` mode while preserving explicit `route` and `conduct` overrides and the fail-closed `contextual-orchestrator-contract-v1` adapter boundary.
