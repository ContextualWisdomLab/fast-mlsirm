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

### Fail-closed NumPy fallback limits

The retained Python fallback is not the unbounded large-scale production backend. Before it allocates dtype-conversion copies, calls `numpy.where`, constructs quadrature nodes, or creates person-by-node and item-by-node iteration grids, it now enforces two deterministic limits:

1. `n_nodes` must be an integer from **1 through 100**. NumPy documents `hermegauss` as requiring a positive degree and states that results have only been tested through degree 100; higher degrees may be problematic. This is an implementation-support boundary, not a claim that 100 nodes are universally necessary or optimal. The node count is validated before even shape-only NumPy coercion of the response operands, so a malformed quadrature configuration cannot force an arbitrary array-like response object to materialize first.
2. The fallback's conservative owned-workspace estimate must not exceed **512 MiB**. For persons `P`, items `I`, and nodes `Q`, the estimate is:

   ```text
   8 × [5PI + 6PQ + 14IQ + 12(P + I + Q)] + [PI + 2I] bytes
   ```

   The first term budgets float64 response/mask conversions, temporary response arithmetic, posterior grids, Newton grids, Newton arithmetic temporaries, and one-dimensional state; the second budgets Boolean state. The estimate intentionally overstates repository-owned NumPy arrays to preserve headroom. It does not claim to equal process RSS and does not include caller-owned input storage or hidden BLAS workspace.

Problems above the cap fail with a deterministic `ValueError` directing the caller to the Rust backend or a smaller response matrix or quadrature rule. Quadrature validation precedes response-array coercion; after shape-only normalization supplies the dimensions required for budgeting, the workspace check still runs before typed conversion and the fallback's large owned arrays. This closes the resource-exhaustion path without truncating data, silently reducing quadrature, or changing the numerical result for accepted problems.

## Statistical interpretation boundary

This refactor changes only the spelling of the final posterior expectation and adds fail-closed resource limits to the NumPy reference/fallback path. It does not change accepted-problem quadrature nodes or weights, missing-data masking, posterior normalization, M-step updates, convergence criteria, or returned transport. Numerical parity of this projection does not by itself establish parameter recovery, model fit, predictive validity, fairness, scoreability, causal interpretation, or high-stakes deployment readiness.

## Verification contract

`tests/test_mmle_eap_projection_contract.py` provides permanent evidence for:

- a one-iteration posterior reconstruction under partial missingness and parity against the previous weighted-sum equation;
- an AST-level source contract requiring the final `theta` assignment to be `posterior @ nodes`, so the explicit broadcast-product expression cannot silently return;
- rejection of invalid and untested quadrature counts before node construction;
- rejection of an enormous zero-stride response view without materializing response-, person-node-, or item-node-sized fallback grids;
- ordering evidence that the workspace cap runs before typed array conversion and `numpy.where` response-grid allocation;
- release-note language that avoids fixed speedup promises and records hardware/BLAS variability; and
- this doctoring record's NumPy semantics, 512 MiB fallback cap, documented quadrature support range, and Rust-primary architecture boundary.

`tests/test_mmle_quadrature_preflight_order.py` separately uses an array-like sentinel that raises on NumPy coercion and requires an invalid quadrature count to raise the package-owned `ValueError` first. That regression pins the fail-closed configuration ordering without allocating a large response object.

Repository-wide CI, security, packaging, coverage, and independent-review gates remain authoritative. This bounded optimization and hardening do not justify weakening any gate.

## Rollback

If a supported NumPy backend demonstrates a correctness defect in the matrix-vector path, revert the single EAP projection to the algebraically equivalent weighted-sum expression and retain the parity regression while the backend-specific issue is isolated. Do not remove the fallback resource limits as part of that rollback. A performance-only regression is not a reason to change statistical semantics; benchmark evidence should identify the affected shapes, layout, hardware, and numerical-library build before any follow-up optimization.

## References

NumPy Developers. (n.d.). *numpy.matmul*. NumPy documentation. Retrieved August 8, 2026, from https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

NumPy Developers. (n.d.). *numpy.polynomial.hermite_e.hermegauss*. NumPy documentation. Retrieved August 8, 2026, from https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite_e.hermegauss.html
