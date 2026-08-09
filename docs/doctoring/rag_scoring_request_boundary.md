# Governed reference-free RAG scoring-request boundary

## Scope

This record governs the provider-neutral request/provenance adapter implemented by `fast_mlsirm.scoring.rag`. The adapter maps one RAG evaluation event onto the existing canonical `ScoringRequest` axes and validates identities, evidence regime, candidate visibility, fingerprints, bounded caller metadata, and response-size evidence. It does **not** retrieve documents, call an LLM, compute a RAG metric, set a pass/fail threshold, persist a hosted workflow, or adjudicate factual truth.

Python owns validation and provenance marshalling at this boundary. Numerical scoring, calibration, likelihood, uncertainty, DIF/invariance, or hierarchical measurement arithmetic introduced later remains Rust-owned under the repository numerical-ownership contract.

## Measurement rationale

Reference-free evaluation is not truth-free evaluation. RAGAS separates RAG quality into constructs such as faithfulness, answer relevance, and context relevance rather than treating a single automated score as world truth. ARES likewise evaluates context relevance, answer faithfulness, and answer relevance with learned judges and explicitly uses human-annotated data to correct evaluation uncertainty through prediction-powered inference. RAGChecker further motivates fine-grained diagnostics across retrieval and generation instead of collapsing heterogeneous failure modes into one undifferentiated score.

The request contract therefore records *what evidence was available* and *which system/run/query/response produced the observation* without claiming that the observation proves factual correctness, absolute retrieval recall, fairness, causal validity, or deployment readiness. Those claims require separate identified validation evidence.

## Identity and provenance contract

The shared scoring axes are reused rather than forked:

- `respondent_id` is the stochastic `system_run_id`;
- `response_id` is the distinct generated-answer identity;
- `task_id` is `query_id`;
- `task_revision_fingerprint` is the exact query-revision fingerprint;
- `task_family_id` is `query_testlet_id`;
- `occasion_id` remains the shared occasion axis; and
- system-configuration identity/fingerprint, retrieval-run fingerprint, query-revision fingerprint, evidence regime, and candidate visibility are package-managed metadata participating in canonical request identity.

System configuration, stochastic run, and generated response are deliberately distinct. This prevents repeated runs under one configuration from collapsing into one observation and preserves later rater/model/occasion variance analysis.

## Content-minimization boundary

Raw query, retrieved-context, answer, and source text are intentionally absent from the builder interface and canonical RAG metadata. Caller metadata is closed to the single `evaluation_split` key in this slice, and that value must be a descriptive two-or-more-token lower-snake identifier such as `offline_holdout`; arbitrary raw content fails closed with the stable non-reflective `invalid_evaluation_split` error.

This is not heuristic content inspection. The contract minimizes ambient content channels by schema and identifier validation. Exact source material may remain in a purpose-authorized host system, but the reusable core artifact prefers opaque identities and fingerprints so scoring evidence can be audited without propagating raw content into logs, model prompts, or release evidence.

## Interpretation boundaries

The evidence regimes (`prompt_only`, `retrieved_context`, `pooled_corpus`, `authoritative_corpus`, `human_anchor`) describe evidence availability, not a validity ladder. In particular, `retrieved_context` does not establish world correctness or complete recall, `authoritative_corpus` does not remove retrieval or judge error, and `human_anchor` does not make an individual judgment infallible. Candidate visibility is recorded separately because evaluator access to candidate identity/content can change the measurement process.

Future RAG metric observations should preserve construct-specific identities and judge/rater provenance rather than silently converting metric output into ground truth. Any consequential threshold, composite score, fairness claim, or deployment decision requires its own validation and governance contract.

## Verification contract

The focused RAG request tests require exact shared-axis mapping, stable enum/fingerprint failures, package-managed provenance, reserved-key rejection, raw-content exclusion, stochastic-run identity separation, and descriptive-identifier validation for `evaluation_split`. The complete repository suite remains authoritative for compatibility with shared scoring, serialization, security, Rust/PyO3, packaging, and release-evidence contracts.

## References

Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024). RAGAs: Automated evaluation of retrieval augmented generation. *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations*, 150–158. https://doi.org/10.18653/v1/2024.eacl-demo.16

Ru, D., Qiu, L., Hu, X., Zhang, T., Shi, P., Chang, S., Cheng, J., Wang, C., Sun, S., Li, H., Zhang, Z., Wang, B., Jiang, J., He, T., Wang, Z., Liu, P., Zhang, Y., & Zhang, Z. (2024). RAGChecker: A fine-grained framework for diagnosing retrieval-augmented generation. *Advances in Neural Information Processing Systems, 37*. https://doi.org/10.52202/079017-0692

Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An automated evaluation framework for retrieval-augmented generation systems. *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, 338–354. https://doi.org/10.18653/v1/2024.naacl-long.20
