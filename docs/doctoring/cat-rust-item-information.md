# CAT Rust-owned item-information doctoring

## Status and scope

This record governs the public `item_information()` boundary in
`python/fast_mlsirm/test_design.py`. The public function keeps Python-side shape,
factor, person/theta, and immutability validation, but the probability and
Fisher-information arithmetic is owned by the compiled Rust scoring core through
`_core.bank_information` / `mlsirm_core::scoring::bank_information_device`.

The supported contract is the existing simple-structure CAT item-information
surface: each item maps to one trait through `factor_id`, and the current
MIRT/MLS2PLM-family predictor semantics and population-mean latent-position
convention are preserved. This change moves numerical ownership without changing
the estimand.

## Numerical and ownership contract

For dichotomous items, the item-information quantity remains the ordinary Fisher
information for the active trait at the requested ability point. Python does not
recompute probabilities or `a^2 P(1-P)` on the production path. It validates and
marshals contiguous immutable inputs, calls the compiled core, validates the
returned shape, and transports the Rust-owned item-information vector.

Missing or incompatible compiled capability fails closed rather than silently
selecting a second Python numerical implementation. The final global
administered-item masking and deterministic maximum policy in `select_cat_item()`
remains a separate ownership slice because the existing Rust `cat_next_item`
implements a different adaptive policy; substituting it here would change public
semantics.

## Verification boundary

The ownership regression replaces `_core.bank_information` with an unmistakable
sentinel and proves that the public function returns the Rust-provided vector
exactly. A separate immutability regression proves caller-supplied `theta` and
`factor_id` arrays are not modified by marshalling. Full package CI also exercises
the existing CAT probability/information and adaptive-test behavior against the
same compiled scoring implementation used by serving.

This evidence establishes one production numerical owner for the public
item-information vector. It does not establish construct validity, fairness,
consequential-decision readiness, or that the remaining final CAT selection and
fixed-form assembly policies are Rust-owned.

## Rollback and follow-up

If the compiled bank-information path is defective, revert to the last verified
Rust implementation or disable the affected public boundary; do not restore a
silent Python probability/information fallback. Issue #629 remains the governing
follow-up for the final global CAT next-item policy and fixed-form assembly
numerical ownership. A future change must preserve the public selection estimand
or document and validate a deliberately new adaptive policy.

## References

Baker, F. B., & Kim, S.-H. (2017). *The basics of item response theory using R*.
Springer. https://doi.org/10.1007/978-3-319-54205-8

Lord, F. M. (1980). *Applications of item response theory to practical testing
problems*. Lawrence Erlbaum Associates.

van der Linden, W. J., & Pashley, P. J. (2010). Item selection and ability
estimation in adaptive testing. In W. J. van der Linden & C. A. W. Glas (Eds.),
*Elements of adaptive testing* (pp. 3–30). Springer.
https://doi.org/10.1007/978-0-387-85461-8_1
