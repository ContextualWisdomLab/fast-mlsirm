# Mokken AISP explicit decision controls

## Status

Implemented on active branch only. Protected `main` remains authoritative until the associated pull request merges through ordinary governance.

## Root cause

The Rust `aisp` kernel already requires the scalability lower bound `c` and nominal significance level `alpha` as explicit inputs. The Python public wrapper, however, supplied `lower_bound=0.3` and `alpha=0.05` automatically. Repository documentation itself described `0.3` as a rule of thumb. Those defaults therefore crossed the boundary from literature context into substantive measurement-policy authority: a caller that supplied only response evidence silently received item-scale admission decisions under hand-selected cutoffs.

## Decision

`mokken_analysis` supplies no scientific decision defaults for `lower_bound` or `alpha`. Both controls must be explicitly supplied by the caller after response evidence passes the existing trust/resource boundary. Omission fails closed before Rust capability discovery. Explicit values, including `0.3` and `0.05`, remain accepted when the owning analysis plan intentionally specifies them; the library does not represent those values as universally valid or infer that their presence establishes validity for a population, instrument, estimand, or use.

The Rust numerical algorithm is unchanged. `fast-mlsirm` continues to own deterministic computation and validation, while the downstream measurement design owns the substantive threshold/significance policy and its preregistration, simulation, sensitivity, multiplicity, or other inferential justification as appropriate to the claim.

## Executable provenance

The branch adds a regression that requires omitted AISP controls to fail closed and verifies that explicit controls still reach the Rust boundary. Existing successful fixtures that previously depended on implicit defaults now state their test controls explicitly. Malformed/hostile response evidence retains precedence, so removing the scientific defaults does not weaken the response trust boundary.

## Product-gap disposition

This closes one concrete no-heuristics gap at the Python-to-Rust Mokken AISP boundary. It does not assert that every Mokken threshold used by downstream products is validated; each downstream product still owns the evidence for its chosen controls. The corresponding entry should be incorporated into `docs/product-technical-gap-baseline.md` when its active single-writer lane is reconciled, without replacing concurrent baseline work.

## References

van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of Statistical Software, 20*(11), 1–19. https://doi.org/10.18637/jss.v020.i11

Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing optimization algorithms for item selection in Mokken scale analysis. *Journal of Classification, 30*(1), 75–99. https://doi.org/10.1007/s00357-013-9122-y
