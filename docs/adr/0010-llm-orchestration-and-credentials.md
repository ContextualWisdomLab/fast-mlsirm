# ADR-0010: LLM orchestration and credential boundary

Status: **Accepted**  
Date: 2026-08-09

## Context

Some `fast-mlsirm` features and validation studies may use LLMs for item generation, semantic screening, artificial-crowd responses or LLM-as-a-Judge experiments. Model calls introduce provider credentials, untrusted outputs, cost/rate limits, non-determinism and possible cross-service coupling. Autonomous development agents and independent review agents also use separate authority and must not share identities casually.

## Decision

### Product/model calls

- Prefer provider-neutral interfaces.
- Every LLM-as-a-Judge call MUST use a contextual-orchestrator-backed adapter. The adapter must declare the versioned `contextual-orchestrator-contract-v1` boundary; `ContextualOrchestratorJudge` rejects an unmarked or direct-provider transport at construction. Do not create a reverse source dependency or write there while its owner loop controls that repository. Every cross-repository call MUST bind the versioned request/result schema or an immutable artifact digest, and the compatibility policy for that contract must be recorded with the caller.
- Treat all model output as untrusted; schema/provenance/semantic validation remains inside the calling workflow.
- Deterministic tests/gates that do not require a model call shall remain executable without model credentials.

### Cross-repository judge contract

The versioned boundary is also a public package contract: fast-mlsirm exports
`CONTEXTUAL_ORCHESTRATOR_CONTRACT_V1` from both `fast_mlsirm.llm_judge` and the
package root with the value `contextual-orchestrator-contract-v1`. The
contextual-orchestrator same-interpreter preflight is authoritative for live
judge readiness; a working injected judge with a missing public contract symbol
is an integration failure, not an available integration. A regression test must
cover the package-root export before a live MLX judge result is treated as
cross-repository evidence.

The two repositories remain separately installable and do not acquire a reverse
source dependency. That boundary is easy to misconfigure: isolated project
virtual environments cannot import the other checkout. A live source checkout
therefore MUST use one interpreter with both packages installed (prefer editable
installs), or explicitly expose both source roots with `PYTHONPATH`, and MUST
run `python -m contextual_orchestrator check-fast-mlsirm` with that interpreter
before judging. A missing cross-import is a fail-closed integration error; it
must not be hidden by a second interpreter, keyword/positional repair, or direct
provider fallback.

### Failure-evidence boundary

Failed binary-threshold calls may expose bounded operational evidence, but they
must not retain raw provider output or exception text: either can contain task,
answer, reference, rubric, or provider diagnostics. Evidence is therefore an
allowlisted record of criterion/threshold identity, call/parse status, stable
failure code, exception type, validated Boolean values, trace count, and usage.
This redaction boundary applies before callers persist or publish
`JudgeFormatError.evidence`; it is not a keyword, positional, or silent-repair
mechanism.

### Credentials

- Model-backed GitHub tests/agents use the existing GitHub Secret `NVIDIA_NIM_API_KEY` when the model call requires NVIDIA NIM.
- `COPILOT_GITHUB_TOKEN` is not used for autonomous development scheduling.
- Review-agent credentials/identities remain distinct from development-agent credentials and are not rewritten merely to simplify automation.
- Secrets are materialized only in the step/path that needs them and must not appear in logs, report payloads, error strings or generated audit identifiers.

### Autonomous development scheduling

GitHub Actions autonomous development uses an immutably pinned OpenCode Agent design when repository automation performs model-backed development work. The scheduler does not manufacture approval, weaken branch protection or make advisory model output merge authority.

### Test-time compute/orchestration

When complex LLM orchestration is used, architecture decisions should compare simple single-model routing with deeper orchestration under comparable budgets. Relevant dimensions include:

- workflow stages;
- task decomposition;
- recursion depth;
- tool/access lists;
- role-specific reasoning effort;
- number/family of model calls;
- ablation of deeper reasoning/orchestration.

Correctness, evidence quality, reproducibility and controllability are more important than minimizing latency when the research task explicitly prioritizes inference quality.

## Consequences

Benefits:

- no provider SDK becomes part of psychometric numerical truth;
- secrets and review authority remain separated;
- model-free deterministic CI stays usable;
- LLM orchestration can evolve without rewriting measurement contracts.

Costs:

- live-model validation requires explicit external service availability;
- orchestration evidence can be expensive;
- some end-to-end tests remain bounded/scheduled rather than always-on PR gates.
- test doubles must explicitly declare the same contract marker; an unmarked
  fake is not evidence of a valid production transport path.

## Alternatives considered

1. **Call one vendor SDK directly from core numerical modules.** Rejected because it couples measurement truth to a provider.
2. **Require model credentials for all tests.** Rejected because deterministic package/scientific gates must not depend on unrelated provider availability.
3. **Use development-agent credentials for independent review.** Rejected because it collapses separation of duties and cannot create a legitimate independent approval.

## Reversal conditions

Supersede if CWL adopts a new organization-wide provider/agent credential architecture that provides equivalent least privilege, auditability, independent review separation and deterministic no-model gates.

## Research traceability

Orchestration-depth policies should be documented against current primary research, including Fugu/Conductor/TRINITY-class work or later stronger evidence, when those policies materially affect a released workflow. These studies inform experiments; they do not automatically mandate deep multi-agent orchestration for every task.
