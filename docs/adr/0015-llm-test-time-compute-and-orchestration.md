# ADR 0015: Evidence-governed LLM test-time compute and orchestration

- **Status:** Proposed
- **Date:** 2026-08-09
- **Decision owners:** fast-mlsirm maintainers
- **Scope:** LLM-backed item generation, rubric screening, artificial-crowd administration, automated scoring, and development agents

## Context

LLM-backed workflows can use one model call, a routed sequence of calls, or a deeper multi-agent process with role separation, decomposition, recursion, access controls, verification, and synthesis. More agents or more tokens do not automatically improve correctness. Additional orchestration can increase correlated error, prompt leakage, cost, non-determinism, and attack surface.

`fast-mlsirm` owns provider-neutral measurement contracts and psychometric evidence. It does not own a hosted provider control plane. `contextual-orchestrator` is the preferred downstream orchestration component and remains a read-only dependency while its dedicated writer loop is active.

## Decision

### Explicit experimental variables

Every material LLM-backed workflow declares and versions at least:

- model and model-family identity;
- provider and endpoint policy;
- prompt-template and rubric fingerprints;
- workflow stages;
- task decomposition;
- recursion or refinement depth;
- worker count and role definitions;
- access lists, tools, and information topology for each role;
- role-specific reasoning effort or compute allowance;
- verification, adjudication, and merge strategy;
- retry and total test-time compute budgets;
- random seed, sampling parameters, and occasion identity where applicable.

These values are provenance, not hidden prompt implementation detail.

### Comparable-budget ablation

A deeper orchestration policy is adopted only after comparison with a single-model or shallow-routing baseline under a comparable total budget. Evaluation emphasizes:

- task correctness and evidence quality;
- calibration and abstention behavior;
- robustness to perturbation, judge family, and prompt occasion;
- diversity versus correlated failure among workers;
- cost and bounded resource use;
- reproducibility and audit completeness;
- security and authority separation.

Latency is reported but is not the primary objective for research-intensive evaluation workflows.

### Role and authority separation

Generator, critic, evidence verifier, psychometric reviewer, security reviewer, and final synthesizer roles may be separated when the separation is empirically useful. Separation never grants merge, release, signing, protection, identity, or data-access authority. Models and workers return hostile untrusted outputs that must pass deterministic schema, provenance, evidence, policy, and psychometric gates.

A model-generated approval, confidence statement, or consensus does not substitute for a GitHub-counted independent human approval, release gate, or high-stakes human decision.

### Scheduling and credentials

LLM-backed tests and autonomous development use GitHub Secret `NVIDIA_NIM_API_KEY`. Scheduled development agents use an immutably pinned OpenCode Agent. `COPILOT_GITHUB_TOKEN` is not used for autonomous development scheduling. Existing independent review-agent identities, credentials, and separation-of-duties contracts are preserved.

Live model conformance tests are bounded and separated from deterministic merge gates unless repository policy explicitly requires the live path and its availability contract. Secrets are materialized only in the actual model-calling step, never in deterministic inventory, validation, or dry-run stages.

### Measurement treatment

LLM outputs used as ratings are fallible observations. Model, prompt, provider, role, occasion, and workflow policy remain separate facets or provenance dimensions. Artificial-crowd responses do not become population truth by volume. Calibration, fit, local dependence, range restriction, DIF, drift, and judge-family sensitivity are assessed before operational use.

## Consequences

### Positive

- Compute allocation is an auditable experimental policy rather than an implicit prompt choice.
- Deeper orchestration must demonstrate incremental value over a simpler baseline.
- Worker roles can vary reasoning effort and access without receiving repository or release authority.
- Model/rater effects can be represented in psychometric calibration and drift analysis.
- Provider-specific implementation remains outside the reusable core.

### Costs and limitations

- Comparable-budget experiments are expensive and may require multiple provider families.
- Provider behavior and model versions can change faster than measurement contracts.
- Deterministic fixtures remain necessary because live conformance alone is not reproducible.
- Research such as Fugu, Conductor, TRINITY, and later test-time-scaling work informs hypotheses; it does not establish universal superiority of one topology.

## Verification obligations

Applicable features require:

1. deterministic provider-neutral request and result schemas;
2. hostile-output and credential-boundary tests;
3. single/shallow versus deeper comparable-budget ablations;
4. provenance for every model, prompt, role, tool, access policy, and occasion;
5. bounded total compute, retry, recursion, and worker counts;
6. independent evidence verification rather than self-asserted correctness;
7. calibration, abstention, local-dependence, DIF, drift, and judge-family analysis;
8. failure-mode tests for unavailable providers, partial worker failure, conflicting judgments, and budget exhaustion;
9. explicit rollback to a simpler or deterministic path.

## Rejected alternatives

1. **Always use the largest possible agent swarm.** Rejected because more agents can amplify correlated error and cost.
2. **Use one model for generation, judging, and final approval without provenance.** Rejected because roles and errors cannot be separated.
3. **Treat majority vote as calibrated truth.** Rejected because shared model-family and prompt effects violate independence.
4. **Place provider SDKs and secrets in the psychometric core.** Rejected because it destroys standalone and modular boundaries.

## Research watch

Primary papers governing a concrete implementation must be committed when redistribution is permitted or cited and summarized when it is not. The doctoring record must include both supportive and counterevidence on test-time scaling, decomposition, verification, and multi-agent error correlation. Published evidence remains descriptive of the studied tasks and models rather than a universal architecture mandate.
