# ADR-0005 — Human and LLM Judges as Fallible Raters

- **Status:** Accepted
- **Date:** 2026-08-09
- **Decision owner:** `fast-mlsirm`
- **Implementation status:** active principle; specific rater/range/drift estimators require their own evidence

## Context

Automated scoring and reference-free RAG evaluation often report correlation or agreement with a human or another model and then treat that value as accuracy or validity. Human raters and LLM judges both exhibit severity, criterion bias, range-use differences, prompt/order effects, drift and stochastic error. A high correlation can coexist with additive/scale bias, range compression, subgroup error or common shortcut contamination.

## Decision

Human, LLM and external scorers are represented as measurement instruments with explicit rater/engine/model/version/prompt/occasion provenance. Raw judge scores are observations, not ground truth.

The reusable scoring layer preserves analytic criteria, scored/abstained/failed/excluded states and exact evidence identity. Appropriate analyses may estimate or report severity, discrimination/consistency, range use, agreement, DIF/subgroup behavior and drift. High-stakes automation requires separate validity/use evidence and human/governance policy.

## Invariants

- Pearson/Spearman correlation is descriptive and never the sole accuracy or interchangeability criterion.
- When true parameters are known, bias/MAE/RMSE/coverage/recovery take priority.
- Human reference ratings are not silently relabelled as true scores.
- Terminal states are not coerced into low ordinal ratings.
- Criterion-level evidence is preserved rather than averaged before calibration.
- LLM provider failures and untrusted outputs do not expose source/private text in stable errors.
- Candidate/model/prompt versions are retained so drift can be detected.

## Alternatives considered

1. Raw mean/majority vote — useful baseline but loses rater effects and uncertainty.
2. One preferred LLM judge as oracle — rejected.
3. Human score as unquestioned gold truth — rejected when rater error is relevant.
4. Shared observation contracts plus psychometric/rater validation — accepted.

## Consequences

Evaluation costs may increase because multiple raters/judges and connected designs are valuable. In return, the product can distinguish disagreement caused by measurement error from systematic rater perspective and can route uncertain or high-impact cases for review rather than fabricating precision.

## Failure / degraded behavior

If the rater/task design is disconnected or too sparse to identify a rater effect, return a design/evidence error or descriptive metrics only. Do not estimate severity from disconnected groups and then compare them as though linked.

If no defensible truth/reference exists, report groundedness/consistency or latent consensus with explicit limits; do not rename it world correctness.

## Security, privacy and AI governance

Model-backed automation uses the approved provider/credential boundary, with NVIDIA NIM/OpenCode where configured. `COPILOT_GITHUB_TOKEN` is not model authentication. Reviewer credentials remain independent of scoring/model execution credentials. NIST AI RMF and NIST AI 600-1 inform risk controls without creating a certification claim.

## Verification

- human/AI exact and adjacent agreement and QWK fixtures;
- severity/range-use/drift designs where implemented;
- shuffled order/prompt perturbation and model-family sensitivity;
- subgroup/DIF evidence;
- simulated rater bias/variance recovery;
- raw-correlation counterexamples demonstrating why agreement/recovery is needed.

## Sources

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Uto, M., & Ueno, M. (2020). A generalized many-facet Rasch model and its Bayesian estimation using Hamiltonian Monte Carlo. *Behaviormetrika, 47*, 469–496. https://doi.org/10.1007/s41237-020-00115-7

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet, 327*(8476), 307–310.

Autio, C., Schwartz, R., Dunietz, J., Jain, S., Stanley, M., Tabassi, E., Hall, P., & Roberts, K. (2024). *Artificial intelligence risk management framework: Generative artificial intelligence profile* (NIST AI 600-1). National Institute of Standards and Technology. https://doi.org/10.6028/NIST.AI.600-1

## Supersession criteria

Supersede only if a more general evidence model can represent human/AI raters, criterion and occasion effects with equal or stronger identification, recovery and auditability while preserving these no-oracle/no-correlation-only invariants.
