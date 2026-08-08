# MMLE EAP matrix-vector projection doctoring

## Status and scope

This record governs the NumPy **reference/fallback** implementation of the unidimensional 2PL MMLE EAP projection in `python/fast_mlsirm/estimators/mmle.py`. Production psychometric arithmetic remains Rust-first; this Python path exists for compatibility, reference validation, and environments where the Rust backend is unavailable. This change does not move likelihood, optimization, scoring, or calibration ownership from Rust into Python.

## Behavioral invariant

After the final E-step, each person has a normalized posterior row `posterior[p, :]` over the Gauss-Hermite quadrature nodes `nodes[:]`. The EAP ability estimate is the posterior-weighted node sum. For a two-dimensional posterior and one-dimensional node vector, NumPy defines `posterior @ nodes` as matrix-vector multiplication and returns one value per posterior row. The result is algebraically equivalent to the previous expression `(posterior * nodes[None, :]).sum(axis=1)`.

The implementation therefore uses:

```python
theta = posterior @ nodes
```

The permanent regression reconstructs one complete first-iteration posterior under partial missingness independently of production posterior helpers, evaluates the previous weighted-sum equation in test code, and requires tight floating-point parity with the public fallback result.

## Resource and performance boundary

The previous spelling explicitly formed an element-wise posterior-shaped broadcast product before reducing it. The matrix-vector spelling avoids constructing that explicit `N × Q` product and expresses the operation directly as a linear-algebra primitive. NumPy documents `@` for ndarrays as `numpy.matmul`; optimized BLAS may be used when available.

No fixed or universal speedup is claimed. Runtime performance remains dependent on dimensions, memory layout, dtype, hardware, thread configuration, and the linked numerical library. `matmul` still produces the required output array and may use implementation-specific workspaces. The contract is therefore **absence of the explicit posterior-shaped broadcast product**, not a zero-allocation claim or a universal latency ratio.

## Statistical interpretation boundary

This refactor changes only the spelling of the final posterior expectation in the NumPy reference/fallback path. It does not change quadrature nodes or weights, missing-data masking, posterior normalization, M-step updates, convergence criteria, or returned transport. Numerical parity of this projection does not by itself establish parameter recovery, model fit, predictive validity, fairness, scoreability, causal interpretation, or high-stakes deployment readiness.

## Verification contract

`tests/test_mmle_eap_projection_contract.py` provides permanent evidence for:

- a one-iteration posterior reconstruction under partial missingness and parity against the previous weighted-sum equation;
- an AST-level source contract requiring the final `theta` assignment to be `posterior @ nodes`, so the explicit broadcast-product expression cannot silently return;
- release-note language that avoids fixed speedup promises and records hardware/BLAS variability; and
- this doctoring record's NumPy semantics and Rust-primary architecture boundary.

Repository-wide CI, security, packaging, coverage, and independent-review gates remain authoritative. This bounded optimization does not justify weakening any gate.

## Rollback

If a supported NumPy backend demonstrates a correctness defect in the matrix-vector path, revert the single EAP projection to the algebraically equivalent weighted-sum expression and retain the parity regression while the backend-specific issue is isolated. A performance-only regression is not a reason to change statistical semantics; benchmark evidence should identify the affected shapes, layout, hardware, and numerical-library build before any follow-up optimization.

## References

NumPy Developers. (n.d.). *numpy.matmul*. NumPy documentation. Retrieved August 8, 2026, from https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
