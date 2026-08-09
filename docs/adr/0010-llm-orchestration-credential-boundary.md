# ADR-0010: Keep LLM orchestration optional and isolate model credentials from review/merge authority

- Status: **Accepted**
- Date: 2026-08-09
- Owner: automation/provider-integration boundary

## Context

Some development, scoring, rubric-generation, and evaluation workflows benefit from live LLMs. The organization also uses independent automated reviewers and guarded merge policies. Reusing reviewer or GitHub credentials as model-provider credentials, or embedding provider-specific orchestration into the numerical core, would couple security authorities and make deterministic scientific gates dependent on an external generative model.

The preferred hosted model credential for development/test automation is `NVIDIA_NIM_API_KEY`. `contextual-orchestrator` is the preferred reusable orchestration integration when its capabilities add value. The accepted research direction uses workload/stage-aware test-time compute allocation between direct model calls and deeper multi-agent decomposition rather than optimizing primarily for latency.

## Decision

1. Live model execution is optional to the reusable core and occurs behind a versioned provider/orchestration protocol.
2. GitHub development/model tests use `NVIDIA_NIM_API_KEY` when a live LLM is genuinely required. `COPILOT_GITHUB_TOKEN` is not used as a development model credential.
3. Existing independent review-agent identities, approval tokens, and merge authorities are not repurposed as LLM generation credentials.
4. Deterministic parsing, provenance validation, psychometric arithmetic, security policy, branch protection, and model-selection/scoreability gates are implemented deterministically and cannot be delegated to an LLM merely because an LLM is available.
5. `contextual-orchestrator` may be used through an explicit versioned integration, but `fast-mlsirm` remains independently installable and no orchestration runtime becomes a hidden import/build dependency.
6. For LLM/orchestration workloads, test-time compute policies may allocate between direct routing and deeper agent decomposition by workflow stage, task decomposition, recursive depth, available approaches, and role-specific reasoning effort. The policy is evidence-driven; speed is not the primary scientific objective.

## Invariants

- A provider/model subprocess does not inherit unrelated GitHub repository-write, reviewer, merge, or OIDC credentials unless a narrowly reviewed workflow explicitly requires and validates such authority.
- Model output cannot approve a PR, turn a failed check into success, change branch protection, or silently change the scientific acceptance target.
- Missing provider credentials fail closed for the model-backed action and do not silently fall back to an unreviewed provider.
- Provider/model id, configuration/version, request identity, and relevant prompt/workflow version are preserved where they affect measurement/audit evidence.
- Offline deterministic fixtures remain available so core contract tests do not require network access.

## Alternatives considered

### Use GitHub Copilot/GitHub Models token for every automation

Rejected. It couples model execution to repository authority and conflicts with the organization's credential policy.

### Embed one provider SDK into the psychometric core

Rejected. It makes provider/network lifecycle a hidden scientific dependency and reduces modularity.

### Let the LLM decide whether numerical/security tests pass

Rejected. Generative interpretation is not a substitute for deterministic evidence.

## Failure and recovery

Provider rate limits, unavailability, malformed output, or credential absence block only the affected model-backed step. The workflow continues with independent executable work where possible. Recovery uses a reviewed provider fallback/orchestrator policy or a later rerun; it never fabricates a successful review/scientific result.

## Compatibility and rollback

Provider/orchestrator integrations are additive adapters over stable core contracts. A provider or model can be removed without changing the numerical/scoring artifact schema when provenance fields remain interpretable. Credential-name changes require a workflow/security migration, not an implicit alias.

## Verification

Workflow/provider tests verify secret names, least privilege, environment stripping, exact model/provider configuration, offline fixtures, fail-closed missing credentials, bounded retries/timeouts, untrusted output validation, and separation from reviewer/merge authority.

## Research basis

The orchestration strategy follows the project's accepted Fugu/Conductor/TRINITY research direction for allocating test-time computation across direct routing and deeper role/task decomposition. Exact paper/version citations belong in the feature-specific doctoring when an orchestration algorithm is implemented or changed; this ADR records the system boundary and credential authority, not one immutable research algorithm.

## Consequences

The system keeps deterministic science and repository governance independent from model availability while retaining a reusable path to stronger LLM-assisted generation/evaluation workflows.
