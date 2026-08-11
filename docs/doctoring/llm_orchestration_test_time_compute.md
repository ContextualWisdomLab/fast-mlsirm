# LLM Orchestration and Test-Time Compute Doctoring

Date reviewed: 2026-08-09  
Scope: LLM-backed assessment, rubric/item generation, scoring/judging, and autonomous-development integration policy for `fast-mlsirm`.

## Decision summary

`fast-mlsirm` does not assume that deeper multi-agent orchestration is intrinsically better than a single capable model. When an LLM-backed feature uses orchestration, compute allocation is an explicit experimental/design variable. At minimum, validation distinguishes:

- single-model routing;
- parallel sampling/diversification;
- sequential revision/reflection;
- verifier/aggregator stages;
- heterogeneous worker/model selection;
- role assignment;
- communication/access topology;
- recursive/deeper orchestration;
- reasoning-effort allocation by role/stage; and
- total test-time compute budget.

Performance claims require comparable-budget ablations where scientifically meaningful. Latency is not the primary objective; correctness, evidence quality, reproducibility, controllability, and bounded failure behavior are.

## Evidence reviewed

### Conductor

Nielsen et al.'s *Learning to Orchestrate Agents in Natural Language with the Conductor* (ICLR 2026; arXiv:2512.04388) trains a relatively small coordinator with reinforcement learning to produce worker-specific instructions and communication topologies over heterogeneous LLM pools. The reported architecture can select itself recursively, creating an adaptive test-time-compute axis. This supports making topology, decomposition, worker selection, and recursion first-class orchestration controls rather than hard-coded workflow assumptions.

### TRINITY

Xu et al.'s *TRINITY: An Evolved LLM Coordinator* (ICLR 2026; arXiv:2512.04695) uses a compact coordinator that selects a worker model and assigns roles such as Thinker, Worker, or Verifier over multiple turns. The coordinator is optimized with an evolutionary strategy and is explicitly budget-sensitive. This supports role-specific model/reasoning policies and learned/dynamic delegation, while not implying that the exact TRINITY coordinator should be embedded into this library.

### Sakana Fugu

Sakana AI describes Fugu as a production multi-agent orchestration system that coordinates pools of frontier models and can use recursive/adaptive coordination. Fugu is treated here as product/operational evidence from its developer, not as an independent peer-reviewed scientific source. The architectural implication is provider/model substitutability behind a stable orchestration boundary, not a requirement to depend on Fugu.

### Test-time scaling for agents

Zhu et al. (2025) systematically study agent test-time scaling strategies including parallel sampling, sequential revision, verification/merging, and rollout diversity. Their results support representing these knobs independently in evaluation instead of treating "more agents" as a single scalar configuration.

### Equal-compute caution

Tran and Kiela (2026) report that single-agent systems can match or outperform multi-agent systems on multi-hop reasoning when thinking-token budgets are controlled. This is important counter-evidence against attributing gains to architecture when a multi-agent condition simply spends more inference compute. `fast-mlsirm` therefore requires comparable-budget or explicitly budget-conditioned ablations when choosing between a single scorer/router and deeper orchestration.

Wunderlich et al. (2026) additionally compare self-consistency, self-refinement, debate, and mixture-of-agents over multiple compute configurations and analyze Pareto efficiency. Their results support reporting quality together with test-time compute rather than collapsing every orchestration design into a single unqualified accuracy figure.

## Architectural requirements derived from the evidence

### ORCH-001 — Provider-neutral execution boundary

The measurement core does not own provider SDKs or credentials. An owning adapter/service receives an immutable task/rubric/scoring contract and returns bounded typed output/provenance.

### ORCH-002 — Explicit stage graph

A multi-stage workflow records at least:

- stage identity/type;
- model/engine identity;
- role;
- input/provenance references;
- allowed tools/access list;
- reasoning-effort/budget parameter where the provider exposes one;
- parent/recursive depth;
- stop/verification outcome; and
- token/call/compute usage where measurable.

### ORCH-003 — Comparable-budget ablation

When claiming that orchestration improves evaluation/generation quality, compare against an appropriate single-model or simpler-scaffold baseline under a defensible compute budget. At minimum distinguish total model calls and token/reasoning budget; where provider APIs make exact compute unavailable, document the observable budget proxy and limitation.

### ORCH-004 — Role-specific reasoning effort

Reasoning effort may differ by role. Examples:

- routing/classification: lower or bounded reasoning when deterministic evidence is sufficient;
- item/evidence generation: higher reasoning where construct/evidence synthesis is required;
- verifier/auditor: independent evidence-focused reasoning; and
- final psychometric/numerical calculation: no LLM reasoning; use deterministic Rust core.

A role label alone does not justify higher effort; ablate it when it materially affects cost/quality.

### ORCH-005 — Recursion and decomposition bounds

Recursive/decomposed workflows require explicit maximum depth, stage/call count, token/time/compute budget, and termination behavior. A child-agent failure must have a stable classification rather than producing unbounded retry/decomposition.

### ORCH-006 — Access-list authority

An orchestrator may select only from predeclared tools/data/model capabilities authorized by the owning application. Model text cannot expand its own access list, repository permissions, secret scope, merge authority, or release authority.

### ORCH-007 — Evidence-preserving aggregation

Aggregation must retain which worker/stage produced which claim/evidence. Majority vote or final synthesis may be an output strategy, but it cannot erase dissent/provenance needed for rater calibration, uncertainty, or audit.

### ORCH-008 — Psychometric separation

LLM workers/judges generate observations, evidence units, candidates, or qualitative review signals. Likelihoods, parameter estimation, IRT/MIRT/facet calibration, factor/model comparison, ranking/scoring kernels, DIF/linking, and numerical uncertainty remain deterministic Rust-owned computation.

### ORCH-009 — Live-test credentials

Repository live model tests and autonomous-development model calls use GitHub Secret `NVIDIA_NIM_API_KEY`. `COPILOT_GITHUB_TOKEN` is not a model execution credential for this project. Existing independent review-agent credential identities/scopes are not repurposed by product tests.

### ORCH-010 — Deterministic versus live gates

Deterministic schema, parser, routing-policy, budget, provenance, and fallback tests run without live model access wherever possible. Live model tests are bounded conformance/quality experiments and cannot be the sole evidence for deterministic contract behavior.

## Example experimental matrix

| Dimension | Example conditions |
|---|---|
| orchestration depth | single call; router+worker; worker+verifier; recursive coordinator |
| worker count | 1; 2; 4; adaptive |
| model heterogeneity | same-family; mixed-family/provider |
| roles | no roles; thinker/worker/verifier; custom evidence roles |
| reasoning effort | fixed; role-specific; adaptive |
| decomposition | none; fixed; coordinator-generated |
| access list | read-only evidence; retrieval; bounded tools |
| budget | matched token/call budget; quality-first bounded budget |
| aggregation | single output; listwise verifier; voting; evidence-preserving synthesis |

Report quality/error/recovery metrics together with call/token/budget evidence. The software should not optimize for speed at the expense of scientific validity, but unbounded compute is also not an acceptable product contract.

## References — APA 7th

Nielsen, S., Cetin, E., Schwendeman, P., Sun, Q., Xu, J., & Tang, Y. (2026). *Learning to orchestrate agents in natural language with the Conductor*. International Conference on Learning Representations. arXiv:2512.04388.

Sakana AI. (2026, April 24). *Sakana Fugu: A multi-agent orchestration system as a foundation model*.

Tran, D., & Kiela, D. (2026). *Single-agent LLMs outperform multi-agent systems on multi-hop reasoning under equal thinking token budgets*. arXiv:2604.02460.

Wunderlich, F. V., Kaesberg, L. B., Wahle, J. P., Ruas, T., & Gipp, B. (2026). Multi-agent reasoning improves compute efficiency: Pareto-optimal test-time scaling. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 4: Student Research Workshop)* (pp. 1–14). Association for Computational Linguistics. https://doi.org/10.18653/v1/2026.acl-srw.1

Xu, J., Sun, Q., Schwendeman, P., Nielsen, S., Cetin, E., & Tang, Y. (2026). *TRINITY: An evolved LLM coordinator*. International Conference on Learning Representations. arXiv:2512.04695.

Zhu, K., Li, H., Wu, S., Xing, T., Ma, D., Tang, X., Liu, M., Yang, J., Liu, J., Jiang, Y. E., Zhang, C., Lin, C., Wang, J., Zhang, G., & Zhou, W. (2025). *Scaling test-time compute for LLM agents*. arXiv:2506.12928.
