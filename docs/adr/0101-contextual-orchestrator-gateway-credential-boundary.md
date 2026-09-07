# ADR-0101: contextual-orchestrator owns provider credentials and model routing

Status: **Accepted**  
Date: 2026-09-07
Supersedes: ADR-0010 for provider-credential and GitHub model-routing policy; the remaining provider-neutral judge, untrusted-output, deterministic-test, and separation-of-review-authority decisions continue here.

## Context

`fast-mlsirm` owns reusable psychometric measurement contracts and numerical kernels. It does not own LLM provider discovery, provider credentials, provider/model fallback policy, or the organization-wide review control plane. Those concerns are owned by `ContextualWisdomLab/contextual-orchestrator` and the central `ContextualWisdomLab/.github` required workflows.

ADR-0010 originally allowed model-backed GitHub tests and agents in this repository to consume `NVIDIA_NIM_API_KEY` directly. That was appropriate to the older operating model, but it now contradicts the organization-wide provider boundary: provider secrets bootstrap contextual-orchestrator; consumers authenticate to the gateway and do not select or hold provider credentials. Central required workflows route model-backed review through the fail-closed `orchestrator/free` pool without leaf-specified provider, model group, or paid fallback.

The repository already enforces a provider-neutral product boundary: `ContextualOrchestratorJudge` requires the public `contextual-orchestrator-contract-v1` marker and rejects unmarked direct transports. This ADR aligns automation credentials with that same ownership direction instead of maintaining a second direct-provider exception in documentation.

At the time of this decision, the public GitHub Releases collection for `ContextualWisdomLab/contextual-orchestrator` is empty. That observation is not converted into fictional release evidence. A new direct runtime/client dependency from `fast-mlsirm` therefore remains fail closed until the owner publishes an immutable, versioned API/client/schema release acceptable to this consumer. Existing organization review remains owned by the central required-workflow boundary rather than copied into this repository.

## Decision

### Product and research model calls

- Every LLM/LLM-as-a-Judge transport used by `fast-mlsirm` goes through a contextual-orchestrator-backed adapter and a versioned contract that the upstream owner publishes.
- `fast-mlsirm` does not embed provider SDK routing, provider credential discovery, provider/model-group policy, or paid fallback selection.
- The existing `contextual-orchestrator-contract-v1` marker remains a fail-closed compatibility guard, not proof that an immutable upstream release exists.
- A new or upgraded external runtime/client/schema dependency must name an immutable released upstream version. A mutable branch, source checkout, or sibling PR head is not a production dependency.
- Model output remains untrusted. Bounded parsing, psychometric projection, provenance validation, uncertainty handling, and rater/evaluation semantics remain inside this repository.

### GitHub Actions model-backed automation

- Leaf workflows do not receive provider API keys for model selection or execution.
- Applicable model-backed organization workflows are consumed through the central required-workflow boundary and use `orchestrator/free` plus the gateway authentication token exposed by that owner path.
- Leaf configuration must not name a provider, provider group, concrete paid model, or paid fallback. Missing required gateway capability fails closed and is repaired in the contextual-orchestrator or central-workflow owner.
- `COPILOT_GITHUB_TOKEN` is not a model execution credential for autonomous development.
- Review-agent identities and credentials remain separate from development-agent identities and do not create approval authority merely because a model produced a verdict.

### Deterministic and scientific evidence

- Deterministic package, numerical, recovery, security, and contract tests stay runnable without model credentials unless their acceptance object is specifically a live model integration.
- Scientific claims continue to require measurement evidence appropriate to the estimand. LLM output is rater evidence, not ground truth.
- Deeper orchestration or test-time compute requires comparable-budget ablation when it materially affects a released evaluation workflow; provider routing itself remains upstream.

## Invariants and acceptance evidence

1. Current governing `fast-mlsirm` architecture/technical requirements do not instruct this repository to consume provider API keys directly.
2. Model-backed GitHub Actions policy names `orchestrator/free`, a gateway token, and the prohibition on leaf provider/model/group/paid-fallback selection.
3. Direct-provider transports remain rejected by the public judge boundary.
4. No mutable contextual-orchestrator branch or sibling PR head becomes a runtime/build dependency.
5. If an immutable upstream release required for a new direct client dependency is absent, the integration remains non-released rather than pinning mutable source.
6. Central required workflows remain foreign-owner controls; their source is not copied into `fast-mlsirm`.

## Consequences and trade-offs

Benefits:

- provider credentials and routing stay with one canonical owner;
- the psychometric core remains independently installable and provider-neutral;
- model-backed Actions cannot silently bypass organization routing/cost policy;
- missing upstream release evidence becomes visible instead of being replaced by a mutable dependency.

Costs:

- a provider outage or missing gateway capability can block the model-backed lane even when a leaf developer has a provider key;
- a new direct product/research client dependency cannot be promoted until contextual-orchestrator publishes an immutable compatible release;
- provider-specific troubleshooting belongs in the owning repository and may require cross-owner coordination.

## Alternatives considered

1. **Keep `NVIDIA_NIM_API_KEY` as a leaf secret.** Rejected because it duplicates provider credential/routing ownership and permits a direct-provider path outside contextual-orchestrator policy.
2. **Allow provider/model fallback in leaf workflow YAML.** Rejected because it splits cost, capability, and failover truth across repositories.
3. **Pin a mutable contextual-orchestrator branch or current `main`.** Rejected because exact source identity is not the same as an immutable released consumer contract and creates a mutable cross-repository production dependency.
4. **Copy the central sidecar/workflow into this repository.** Rejected because `.github` and contextual-orchestrator own those control-plane behaviors.

## Failure, recovery, compatibility, and rollback

Missing gateway authentication, missing `orchestrator/free` capability, incompatible released schema/client, or absent immutable release is a typed fail-closed integration state. Recovery occurs in the canonical owner, followed by an immutable release and an explicit consumer adoption here. The leaf must not repair availability by adding a direct provider key, selecting a paid model, weakening a required workflow, or pinning a mutable owner branch.

Rollback of a future released client adoption returns to the previously accepted immutable release or disables the model-backed integration. It does not restore ADR-0010's direct-provider credential policy.

## Evidence

- `ContextualWisdomLab/.github` protected `main@78a4937c684a54ca8e415822c913742f41c6efc4` records central review routing through contextual-orchestrator and `orchestrator/free`.
- `ContextualWisdomLab/contextual-orchestrator` GitHub Releases collection was empty when rechecked on 2026-09-07; this ADR therefore records a release prerequisite rather than claiming one exists.
- `fast_mlsirm.llm_judge.ContextualOrchestratorJudge` already requires `contextual-orchestrator-contract-v1`, providing a code-level direct-provider rejection boundary that this credential decision aligns with.
