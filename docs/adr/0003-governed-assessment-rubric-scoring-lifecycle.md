# ADR-0003: Governed Assessment, Rubric, Scoring, and Item Lifecycle

- **Status:** accepted
- **Date:** 2026-08-09

## Context

The project now spans automated scoring, reference-free RAG evaluation, generated rubric/item workflows, enterprise issue measurement, and human/AI rater calibration. If each domain invents its own assessment/rubric/rater schemas, the numerical core cannot provide reproducible calibration or version linking. If an LLM-generated score or candidate item is trusted because it conforms to JSON, provider errors and evaluator bias become part of the measurement scale unnoticed.

## Decision

Maintain a single reusable contract family centered on assessment, construct, rubric, scoring, observation, evidence, and calibration identity. Domain adapters may specialize these contracts but must not redefine equivalent durable concepts.

The target lifecycle is:

```text
Assessment/Rubric specification
→ Blueprint
→ Bounded generation contract
→ Untrusted candidate output
→ Structural/evidence/semantic screening
→ Human/AI/artificial-crowd pilot observations
→ Rust psychometric calibration
→ Governed item/criterion bank
→ Monitoring/linking/quarantine/retirement
→ New rubric revision
```

Operational versions are immutable. A semantic change creates a new revision and preserves linkability/provenance.

## Judge decision

Human and AI judges are **raters**, not truth by definition. Observations retain rater/model/version/prompt/occasion identity. Where the model and design identify them, severity, criterion-specific bias, discrimination/consistency, range use, and drift are separated from target quality.

Reference-free evaluation must state its evidence regime. Groundedness against retrieved context must not be called world correctness. Completeness/recall claims require an explicit evidence universe or anchor.

## Generated-item trust boundary

Provider output is hostile/untrusted input until validated. Parsing and screening must be bounded and fail closed for malformed schemas, duplicate keys, non-finite values, invalid answer-key references, score-order mismatches, provenance replay, nonexistent evidence/source IDs, contradictory response-format semantics, and other governed violations.

Structural validity is not psychometric validity. Candidate items still require content/evidence screening and empirical calibration before operational use.

## Benchmark vs discovery

- Benchmark/evaluation rubrics should be candidate-blind where possible.
- Candidate-aware criterion discovery is permitted for diagnostics/training only when discovery and scoring samples are separated (for example, cross-fitting) or the result is explicitly labeled exploratory/non-comparative.

## Policy criticality vs discrimination

Psychometric discrimination measures how well an item separates latent quality; it is not business/safety criticality. A rare but catastrophic violation can be a conjunctive policy gate even if its IRT discrimination is low.

## Consequences

Positive:

- RAG, essay, enterprise-issue, and future domains share one reproducible measurement substrate;
- rubric/item versioning and calibration become linkable;
- evaluator bias and provider trust are explicit;
- downstream hosted products can persist these contracts without owning scientific logic.

Costs:

- more explicit schemas and lifecycle states;
- provider adapters must preserve exact provenance and cannot return unconstrained free-form scores as operational evidence.

## References

- Hashemi, H., Eisner, J., Rosset, C., Van Durme, B., & Kedzie, C. (2024). LLM-Rubric: A multidimensional, calibrated approach to automated evaluation of natural language texts. *Proceedings of ACL 2024*, 13806–13834. https://doi.org/10.18653/v1/2024.acl-long.745
- Shankar, S., Zamfirescu-Pereira, J. D., Hartmann, B., Parameswaran, A. G., & Arawjo, I. (2024). Who validates the validators? Aligning LLM-assisted evaluation of LLM outputs with human preferences. *Proceedings of UIST 2024*, 1–14. https://doi.org/10.1145/3654777.3676450
- Williamson, D. M., Xi, X., & Breyer, F. J. (2012). A framework for evaluation and use of automated scoring. *Educational Measurement: Issues and Practice, 31*(1), 2–13. https://doi.org/10.1111/j.1745-3992.2011.00223.x
