# Doctoring record: fallback MMLE EAP matrix-vector projection

## Decision

The NumPy reference/fallback MMLE implementation computes each respondent's expected-a-posteriori ability with the dense matrix-vector product

\[
\widehat{\theta}_p = \sum_q \pi_{pq} x_q,
\]

implemented as `posterior @ nodes`.

This replaces an element-wise broadcast followed by an axis reduction. The statistical quantity, quadrature nodes, posterior weights, missing-data handling, estimator initialization, M-step, stopping rule, and returned transport remain unchanged.

## Architectural boundary

The repository's resolved production backend remains Rust. This change optimizes the explicitly retained NumPy reference/fallback path; it does not move production psychometric arithmetic from Rust into Python and does not introduce a second estimator contract.

For a posterior matrix of shape `(n_persons, n_nodes)`, the former expression materialized an element-wise product with the same shape before reduction. NumPy documents `matmul` and the `@` operator as matrix-product operations and notes that optimized BLAS is used when possible. Exact performance depends on array shape, layout, linked numerical libraries, hardware, and runtime conditions, so this change makes no universal speedup claim.

## Verification contract

`tests/test_mmle_fallback_eap_matmul.py`:

- reconstructs the one-iteration posterior independently for a realistic partially observed response matrix;
- computes the former weighted-sum reference explicitly;
- requires the public fallback result to agree at tight floating-point tolerance; and
- pins the allocation-bounded matrix-vector source path so the broadcast temporary is not silently restored.

The complete repository CI remains authoritative for Rust/PyO3 tests, fallback behavior, package acceptance, GPU no-skip evidence, fuzzing, security scans, and the production coverage/docstring gates.

## Interpretation boundary

Numerical parity of this projection does not establish parameter recovery, model fit, global optimality, construct validity, fairness, or operational readiness. Those claims require the repository's separate simulation, recovery, validation, and governance evidence.

## References

Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters: Application of an EM algorithm. *Psychometrika, 46*(4), 443–459. https://doi.org/10.1007/BF02293801

NumPy Developers. (2026). *numpy.matmul—NumPy v2.5 manual*. https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
