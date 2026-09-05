# Mokken AISP explicit decision controls

## Status

Implemented on active branch only. Protected `main` remains authoritative until the associated pull request merges through ordinary governance.

## Root cause

The Rust `aisp` kernel already requires the scalability lower bound `c` and nominal significance level `alpha` as explicit inputs. The Python public wrapper, however, supplied `lower_bound=0.3` and `alpha=0.05` automatically. Repository documentation itself described `0.3` as a rule of thumb. Those defaults therefore crossed the boundary from literature context into substantive measurement-policy authority: a caller that supplied only response evidence silently received item-scale admission decisions under hand-selected cutoffs.

## Decision

`mokken_analysis` supplies no scientific decision defaults for `lower_bound` or `alpha`. Both controls must be explicitly supplied by the caller after response evidence passes the existing trust/resource boundary. Omission fails closed before Rust capability discovery. Explicit values, including `0.3` and `0.05`, remain accepted when the owning analysis plan intentionally specifies them; the library does not represent those values as universally valid or infer that their presence establishes validity for a population, instrument, estimand, or use.

The Rust numerical algorithm is unchanged. `fast-mlsirm` continues to own deterministic computation and validation, while the downstream measurement design owns the substantive threshold/significance policy and its preregistration, simulation, sensitivity, multiplicity, or other inferential justification as appropriate to the claim.

## Executable provenance

The branch adds a regression that verifies calls with omitted AISP controls fail closed and that explicit controls still reach the Rust boundary. Existing successful fixtures that previously depended on implicit defaults now state their test controls explicitly. Malformed/hostile response evidence retains precedence, so removing the scientific defaults does not weaken the response trust boundary.

On 2026-09-02, exact-head repair run `33573151688` completed successfully after moving the one-shot writer to the available `ubuntu-slim` pool and explicitly provisioning Python, Rust, hash-locked CI requirements, and the editable native package. It updated the five repository-owned successful/core callers that still relied on implicit AISP controls and self-removed its temporary workflow/trigger before the non-force push. The resulting repaired head was `a6dc0e2ddeee80c74ded90cee62d71117f491303`. Pull-request workflow attempts generated from that bot-authored repair head were recorded by GitHub as `action_required` with no jobs, so this traceability commit is intentionally made through the ordinary owner write path to obtain successor-head admission evidence without transferring predecessor results.

## Product-gap disposition

This closes one concrete no-heuristics gap at the Python-to-Rust Mokken AISP boundary. It does not assert that every Mokken threshold used by downstream products is validated; each downstream product still owns the evidence for its chosen controls. The corresponding entry should be incorporated into `docs/product-technical-gap-baseline.md` when its active single-writer lane is reconciled, without replacing concurrent baseline work.

## References and relevance

van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of Statistical Software, 20*(11), 1–19. https://doi.org/10.18637/jss.v020.i11

Relevant finding: van der Ark documents the Mokken scalability coefficients, sample statistics, and AISP implementation exposed by the `mokken` R package. The paper establishes that the lower-bound and significance controls are inputs to the selection procedure; it does not identify one universal pair of values as valid for every instrument, population, estimand, or deployment. This repository therefore uses it to ground the algorithm, not to manufacture an application-wide default.

Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing optimization algorithms for item selection in Mokken scale analysis. *Journal of Classification, 30*(1), 75–99. https://doi.org/10.1007/s00357-013-9122-y

Relevant finding: Straat et al. compare AISP search behavior across lower-bound choices and optimization procedures, showing that the control participates in the item-selection estimand rather than being a transport or formatting constant. Their study supplies methodological context for choosing and evaluating a lower bound; it is not evidence that this library can silently impose one value on every downstream analysis. Accordingly, the wrapper requires the owning analysis plan to state its controls explicitly.
