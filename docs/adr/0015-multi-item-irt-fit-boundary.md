# ADR-0015: Multi-item IRT fit boundary and experiment readiness

Status: **Proposed**
Date: 2026-08-13
Supersedes: none
Superseded by: none

## Context

An IRT result is an item response matrix, not a scalar judge score. The
cross-component contract already requires at least two dichotomous or two
polytomous item columns, but several public numerical fitters still accepted a
one-column matrix before delegating to Rust. That let a caller bypass the
integration contract by calling a fitter directly. Low-level numerical and
diagnostic primitives may still be useful with one item; an interpretable IRT
experiment is not.

LLM-judge outputs add a second risk. A matrix can have the right shape while
having too few persons, an item with almost no observations, a constant item,
or insufficient factor anchors. Such a fit must not be reported as stable
evidence. These checks are measurement gates, not keyword or lexical
judgments, and all LLM-as-a-Judge calls remain routed through
`contextual-orchestrator`.

## Decision

1. The shared `validate_irt_response_matrix` contract is authoritative for
   cross-component data and public IRT model fitters. It requires a 2-D
   persons-by-items matrix with at least two item columns, finite integer
   observations, binary values for dichotomous items, and an explicit bounded
   category count for polytomous items.
2. Public binary and polytomous fitters call that validator after their
   family-specific missing-value normalization and before native computation.
   This includes MLSIRM, 2PL, MH-RM, GRM, GPCM, nominal, RSM, unidimensional
   polytomous, latent-space polytomous, and nominal-polytomous entry points.
3. Low-level kernels, item-level scoring, and deliberately diagnostic
   primitives remain exempt when their own documentation says one-item input
   is meaningful. They must not be presented as an IRT experiment or used to
   manufacture an IRT row.
4. `validate_irt_experiment_readiness` is the pre-interpretation gate. Its
   defaults require at least five persons, three observed responses per item,
   two observed values per item, and, when factor labels are supplied, two
   items per factor anchor. Controls and factor containers are validated
   before counting; strings, booleans, malformed arrays, and unhashable labels
   fail closed.
5. A failed shape or readiness gate is retained as a failed comparison and
   cannot be repaired by keyword matching, positional category repair, silent
   item dropping, or a blind retry. `LLMJudgeResult.to_irt_row` continues to
   require multiple criteria.

## Invariants and acceptance evidence

- `tests/test_irt_contract.py` covers the shared matrix domain, readiness
  controls, factor-container boundaries, and representative public fitter
  rejection of one-item input.
- The targeted IRT/judge/security suite passes after the public-fit boundary
  calls are enabled.
- Each benchmark records item count, category count, person count, observed
  count per item, factor coverage, parse status, provider readiness, and model
  trace metadata before reporting fit or bias.
- A polytomous K-sweep uses explicit K values and preserves category occupancy;
  it does not infer a positive bias from a single K or from keyword matches.

## Consequences and trade-offs

- Direct callers with a one-item matrix receive an early, package-owned
  `ValueError` instead of a native fit or an uninterpretable estimate.
- Small pilot data can still be inspected through explicitly diagnostic APIs,
  but production claims must either meet the readiness defaults or record an
  explicit, justified override in the benchmark artifact.
- The shared validator is called after each model's existing missing-value
  normalization. This preserves the package's established NaN/negative
  missing conventions without duplicating native numerical logic.

## Alternatives considered

### Enforce the rule only in the LLM judge

Rejected: direct numerical callers could still bypass the contract, and the
same measurement boundary would behave differently depending on its producer.

### Reject one-item input in every low-level Rust kernel

Rejected: item-level diagnostics and numerical unit tests have legitimate
single-item uses. The experiment boundary, not every primitive, is the right
scope.

### Convert a scalar into a synthetic second item

Rejected: this creates pseudo-replication and invalid information for IRT.

### Use keyword or category-position matching after an LLM parse failure

Rejected: it is not a measurement model, is vulnerable to prompt/output
format changes, and violates the fail-closed judge contract.

## Failure, migration, and follow-up register

| Finding | Direction | State |
| --- | --- | --- |
| Older callers may pass one item to a public fitter | Build a real multi-item matrix or use an explicitly diagnostic API; do not pad or duplicate a column | Implemented on this branch |
| Shape validation can pass a matrix too small for stable interpretation | Call `validate_irt_experiment_readiness` before benchmark fit and store the rejection reason | Required next |
| A declared polytomous category can be absent or severely imbalanced | Preserve per-item category occupancy and fail or mark the fit non-interpretable before model comparison | Required next |
| Factor labels may not provide two anchors per dimension | Pass factor IDs to readiness validation and require an explicit design exception for exploratory diagnostics | Implemented in validator; benchmark wiring required |
| K/order/framing perturbations can change judge scores | Use randomized paired perturbations, human/gold anchors, category occupancy, and parse/provider denominators; never infer a universal positive-K law | Required next |
| A small local judge model returned a near-schema direct response with `rationale` as an object and criterion scores nested under it, so strict parsing rejected an otherwise usable judgment | Include a complete direct JSON schema example with criterion IDs and explicitly type `rationale` as a string; keep exact-schema parsing and fail-closed behavior | Implemented on current local head; exact-head review follow-up required |
| The same local model then returned all required fields but still used an object for `rationale` | State the non-object rationale boundary explicitly and keep criterion explanations out of that field; do not coerce the object or keyword-match a replacement | Implemented on current local head; exact-head review follow-up required |
| Current Zotero 9.0.6 Local API is read-only | Revalidate records through the local API; perform file writes only after an authorized Zotero 10+ local-write path or authorized Web API is available | Required follow-up |

## References

- Samejima, F. (1969), *Estimation of latent ability using a response pattern
  of graded scores*, https://doi.org/10.1007/BF03372160.
- Li, Q., et al. (2025), *Evaluating Scoring Bias in LLM-as-a-Judge*,
  https://arxiv.org/abs/2506.22316.
- Pezeshkpour, P., & Hruschka, E. (2024), *Large Language Models Sensitivity
  to The Order of Options in Multiple-Choice Questions*,
  https://aclanthology.org/2024.findings-naacl.130/.
- Jones, W. P., & Loe, S. A. (2013), *Optimal Number of Questionnaire Response
  Categories: More May Not Be Better*,
  https://doi.org/10.1177/2158244013489691.
