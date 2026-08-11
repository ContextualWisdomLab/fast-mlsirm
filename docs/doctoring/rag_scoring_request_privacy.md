# Governed RAG scoring request and perturbation-anchor doctoring

## Standards and research basis

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations* (pp. 150–158). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.eacl-demo.16

National Institute of Standards and Technology. (2020). *Security and privacy controls for information systems and organizations* (NIST SP 800-53 Rev. 5). https://doi.org/10.6028/NIST.SP.800-53r5

Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An automated evaluation framework for retrieval-augmented generation systems. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)* (pp. 338–354). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-long.20

Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x

## Privacy and provenance rationale

Reference-free RAG evaluation records provenance identities (system configuration, retrieval run, query revision fingerprints) without embedding raw query, context, answer, or source text. Managed identifiers must reject free-form content so the canonical scoring artifact cannot become a side channel. The host application may retain source material only under its own purpose-bound security, privacy, retention, and authorization controls.

## Controlled perturbation anchors

`RAGPerturbationAnchor` preregisters a finite construct-specific expectation against two distinct governed scoring-request fingerprints. The package currently represents these hypotheses:

- unsupported claim or explicit contradiction -> `grounded_generation` decreases;
- irrelevant context -> `retrieval_relevance` decreases;
- required-evidence removal -> `coverage_or_completeness_proxy` decreases;
- citation-target swap -> `citation_attribution` decreases;
- semantic query paraphrase or style-only rewrite -> `robustness` remains approximately invariant; and
- unanswerable query -> `answerability_and_abstention` increases.

These are controlled-validation expectations, not observed scores, statistical tests, causal effects, ground-truth labels, or evidence that a particular judge can identify world correctness. Context-groundedness remains conditional on the declared evidence regime. A successful anchor regression therefore demonstrates contract identity and preregistered direction semantics only; empirical validation still requires governed observations, calibrated evaluator/rater evidence, subgroup/DIF and drift analysis where applicable, and recovery or human-audit evidence commensurate with the intended interpretation.

## Numerical and ownership boundary

The anchor layer performs no likelihood, calibration, thresholding, uncertainty estimation, scoring aggregation, retrieval, provider call, or truth adjudication. New production psychometric/numerical arithmetic remains Rust-owned with model-appropriate convergence and recovery evidence. Human, rule, AI, and LLM evaluator outputs remain fallible observations rather than truth by identity.

## Implementation

`python/fast_mlsirm/scoring/rag.py`:

- `build_rag_scoring_request` validates `system_configuration_id` via `descriptive_identifier` and allowlists caller metadata;
- `RAGPerturbationKind` and `RAGPerturbationDirection` close the perturbation vocabulary;
- `RAGPerturbationAnchor` derives package-owned construct/direction semantics, rejects no-op baseline/perturbed request pairs, and content-addresses immutable anchor content; and
- `build_rag_perturbation_anchor` exposes only governed identities and perturbation kind, with no raw-content parameters.

`tests/test_scoring_rag_perturbation_anchors.py` binds the finite mapping, deterministic fingerprints, fail-closed malformed/unknown inputs, distinct request identities, and source-free public signature.

This bounded contract advances issue #607 stage-2 validation infrastructure. It does not by itself establish RAG system validity, a scalar quality score, evaluator interchangeability, retrieval recall, world correctness, calibration, DIF invariance, or release readiness.
