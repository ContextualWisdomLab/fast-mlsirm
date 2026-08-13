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
`contextual-orchestrator`. The readiness helper was initially only exercised
by unit tests, so the CLI/benchmark path could still reach native fitting
without this gate.

## Decision

1. The shared `validate_irt_response_matrix` contract is authoritative for
   cross-component data and public IRT model fitters. It requires a 2-D
   persons-by-items matrix with at least two item columns, finite integer
   observations, binary values for dichotomous items, and an explicit bounded
   category count for polytomous items.
2. Every public binary and polytomous estimator entry point calls the shared
   response-matrix validator after its family-specific missing-value
   normalization and before native computation. This includes MLSIRM, 2PL,
   MH-RM, GRM, GPCM, nominal, RSM, unidimensional polytomous, latent-space
   polytomous, and nominal-polytomous entry points.
3. `fit_irt_experiment` is the production/benchmark boundary. It calls
   `validate_irt_experiment_readiness`, accepts scalar factor IDs or per-item
   factor memberships for anchor coverage, and invokes the selected numerical
   fitter only after the gate passes. The CLI `fit` command uses this boundary;
   the CLI `diagnose-dimensions` command validates its source matrix and every
   cross-validation training fold through the same boundary. Direct numerical
   fitters and the default low-level diagnostic mode remain available for
   explicitly diagnostic work and must not be presented as stable experiment
   evidence.
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
  controls, factor-membership boundaries, representative public fitter
  rejection of one-item input, and native-call prevention at
  `fit_irt_experiment`.
- Every public estimator entry point listed in the decision validates the
  multi-item response shape before native-core loading; the production-boundary
  missing-mask regression proves that zero-filled missing cells cannot satisfy
  the observation minimum.
- Production dimension diagnostics reject an unready source matrix before fold
  construction and reject an unready training fold before invoking `fit`.
- The targeted IRT/judge/security suite passes with the CLI production path
  behind `fit_irt_experiment`.
- Each benchmark records item count, category count, person count, observed
  count per item, factor coverage, parse status, provider readiness, and model
  trace metadata before reporting fit or bias.
- A polytomous K-sweep uses explicit K values and preserves category occupancy;
  it does not infer a positive bias from a single K or from keyword matches.

## Consequences and trade-offs

- Direct callers with a one-item matrix receive an early, package-owned
  `ValueError` instead of a native fit or an uninterpretable estimate.
- Small pilot data can still be inspected through numerical/diagnostic APIs,
  but the CLI and benchmark boundary rejects it unless it meets the readiness
  defaults; no silent override is provided.
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
| Shape validation could pass a matrix too small for stable interpretation because the readiness helper had no production call site | Add `fit_irt_experiment` as the production/benchmark gate, preserve missing cells as NaN, accept factor memberships for confirmatory coverage, and keep the numerical fitter behind the gate | Implemented on current head; exact-head review follow-up required |
| `diagnose-dimensions` could bypass the production gate inside its cross-validation loop | Validate the source matrix before fold construction and pass every NaN-preserving training fold through `fit_irt_experiment`; keep the default low-level diagnostic mode explicit | Implemented on current head; exact-head review follow-up required |
| A declared polytomous category can be absent or severely imbalanced | Preserve per-item category occupancy and fail or mark the fit non-interpretable before model comparison | Required next |
| Factor labels may not provide two anchors per dimension | Pass scalar factor IDs or per-item factor memberships through `fit_irt_experiment` and require an explicit design exception for exploratory diagnostics | Implemented on current head; exact-head review follow-up required |
| K/order/framing perturbations can change judge scores | Use randomized paired perturbations, human/gold anchors, category occupancy, and parse/provider denominators; never infer a universal positive-K law | Required next |
| A small local judge model returned a near-schema direct response with `rationale` as an object and criterion scores nested under it, so strict parsing rejected an otherwise usable judgment | Include a complete direct JSON schema example with criterion IDs and explicitly type `rationale` as a string; keep exact-schema parsing and fail-closed behavior | Implemented on current local head; exact-head review follow-up required |
| The same local model then returned all required fields but still used an object for `rationale` | State the non-object rationale boundary explicitly and keep criterion explanations out of that field; do not coerce the object or keyword-match a replacement | Implemented on current local head; exact-head review follow-up required |
| Current Zotero 9.0.6 Local API is read-only | Revalidate records through the local API; perform file writes only after an authorized Zotero 10+ local-write path or authorized Web API is available | Required follow-up |
| The fast-mlsirm default-branch ruleset allowed zero approvals, stale reviews to survive a push, and unresolved review threads | Require one independent approval, dismiss stale approvals on every push, require approval after the last push, and require resolved review threads while preserving the existing merge methods and no-force-push rules | Protected ruleset updated on 2026-08-13; exact-head CI/review evidence remains required |

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
