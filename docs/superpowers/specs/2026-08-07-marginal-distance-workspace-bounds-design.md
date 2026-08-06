# Bounded Marginal Distance Workspaces Design

## Status

Approved for autonomous implementation under issue #563. This is a bounded reliability and performance slice for the NumPy reference/fallback MMLE path. It does not create a second production estimator and does not change the Rust-first product boundary.

## Problem

The marginal latent-space reference path controls the dominant person/item/quadrature working array but does not model every Euclidean-distance allocation. In particular, a broadcast subtraction of item positions and latent-space nodes can materialize an `n_items × n_x × latent_dim` array before reduction. Replacing the reduction with `numpy.einsum` removes only a later square-product temporary; it does not bound or remove the broadcast workspace itself.

A second problem is unit ambiguity. The existing dominant gate is an **element-count** ceiling. Reusing that ceiling for a float64 distance matrix would authorize roughly 800 MB at 100,000,000 elements, so it would not solve the issue's hundreds-of-MiB availability risk. Distance workspaces therefore require a separate, explicit byte budget.

## Selected approach

Use one shared pairwise-distance helper based on the squared-norm identity

\[
\lVert z_i-x_j\rVert^2
=
\lVert z_i\rVert^2+
\lVert x_j\rVert^2-
2z_i^\top x_j.
\]

The helper returns only the required `n_items × n_x` float64 result and never materializes an `n_items × n_x × latent_dim` difference tensor. The matrix-multiplication result is reused in place:

1. compute `left @ right.T` once;
2. multiply that output by `-2` in place;
3. add left and right squared-row norms;
4. clamp negative round-off to zero with `out=`;
5. add `eps_distance`; and
6. take the square root with `out=`.

Small row-norm vectors are permitted. Multiple simultaneous pairwise matrices are not.

## Resource contract

Add a private named ceiling:

```python
MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024
```

The 128 MiB default is an initial conservative fallback-path limit, not a universal hardware capacity claim. It must be accompanied by representative compatibility and peak-memory evidence before merge. Larger workloads should use the Rust production backend rather than silently raising this fallback budget.

Use checked division before multiplication so malformed or astronomically large Python integers cannot create an unbounded intermediate product. Every active distance allocation is calculated in bytes using the actual float64 item size:

- pairwise result: `n_items × n_x × 8`;
- intentional item-local derivative workspace: `n_x × latent_dim × 8`;
- direct-helper left and right matrices, which must already be finite float64 arrays and therefore require no hidden dtype-conversion copy.

The dominant EM working-set ceiling remains a separate element-count contract and is checked through the same safe integer-product primitive. The two limits must not be conflated.

## Interfaces

Add private helpers in `python/fast_mlsirm/estimators/marginal.py`:

```python
def _checked_marginal_workspace_bytes(
    label: str,
    *dimensions: int,
    itemsize: int,
    limit_bytes: int | None = None,
) -> int:
    """Return a bounded byte product or fail before unbounded multiplication."""


def _validate_marginal_distance_workspaces(
    *,
    n_items: int,
    n_x: int,
    latent_dim: int,
    uses_space: bool,
) -> None:
    """Validate actual Euclidean-distance workspaces before node allocation."""


def _pairwise_euclidean_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    eps_distance: float,
) -> np.ndarray:
    """Return bounded float64 pairwise distances without a 3-D tensor."""
```

`uses_space` is an exact Boolean. The direct helper accepts only two-dimensional, finite float64 arrays with the same positive latent width; this prevents an unreported conversion copy before BLAS. Names may be refined only if tests, documentation, and callers use one consistent contract.

## Data flow

1. Convert and validate response/factor inputs as today.
2. Normalize model name, latent dimension, quadrature rule, and exact node count without allocating latent nodes.
3. Derive `free_alpha` and `uses_space`.
4. Apply the dominant EM element-count gate through checked products.
5. If the selected model uses latent space, apply the distance byte gates before `_xi_nodes` or any distance-related allocation.
6. Construct quadrature/node arrays.
7. Use the shared pairwise helper in table construction, candidate predictors, tau updates, and the covariate update.
8. Retain only the guarded `n_x × latent_dim` difference required by the zeta derivative.
9. Preserve existing likelihood, gradient, stopping, missingness, multigroup, multilevel, zero-inflation, and scoring behavior.

## Error behavior

- Boolean, negative, non-integral, or malformed dimensions, item sizes, limits, and `uses_space` flags fail with stable package-owned `ValueError` messages.
- A byte product above the configured ceiling fails before NumPy allocates nodes or the relevant matrix.
- Shape mismatch, non-finite matrices, non-float64 inputs, non-finite/non-positive epsilon, and invalid latent widths fail before BLAS.
- Moderate finite arrays produce finite distances. Small negative squared-distance round-off is clamped to zero before adding epsilon.
- Errors contain no responses, source text, or provider-controlled values.

## Testing

RED→GREEN regressions cover:

1. tight equivalence to the explicit broadcast equation;
2. zero-distance and large-coordinate round-off cases;
3. non-finite input rejection before matrix multiplication;
4. exact float64 byte accounting;
5. oversized pairwise and derivative workspaces rejected without allocation;
6. malformed dimensions, byte limits, item sizes, and Boolean flags;
7. checked failure for astronomically large Python integer dimensions;
8. an otherwise-valid public estimator request rejected before `_xi_nodes`;
9. in-place use of the single matmul output and permanent absence of the 3-D broadcast pattern;
10. realistic partially observed and covariate estimator parity at tight tolerances;
11. complete added-production statement/branch coverage and public docstrings; and
12. exact-head Python, Rust/PyO3, package, explicit GPU-no-skip, fuzz, Security Scan, and SAST evidence.

Source inspection is supplementary. Numerical and allocation evidence remains authoritative.

## Performance and compatibility evidence

Before Ready status, record:

- representative `n_items`, `n_x`, `latent_dim`, and missingness;
- dtype and contiguous-layout assumptions;
- Python, NumPy, linked BLAS, operating system, processor, and memory;
- warm-up and repetition counts;
- elapsed distribution;
- peak resident or traced Python allocation;
- parity against the former equation; and
- the largest repository test/fixture dimensions accepted by the 128 MiB ceiling.

The change may claim removal of one specific 3-D broadcast and a bounded float64 pairwise output. It may not claim a universal speedup or universal memory suitability.

## Alternatives rejected

### Reuse the 100,000,000-element EM ceiling

Rejected. That value represents approximately 800 MB for float64 distance output and is not a byte-level availability contract.

### Keep the broadcast and add only `einsum`

Rejected. `einsum` can avoid an elementwise-square temporary but does not remove the three-dimensional subtraction.

### Build several two-dimensional intermediates

Rejected. It lowers asymptotic dimensionality but can still multiply peak memory. The selected helper mutates one pairwise output in place.

### Chunk every distance computation immediately

Deferred. Chunking adds loop and tuning complexity. It remains a follow-up if the justified 128 MiB output ceiling excludes required reference workloads.

### Move production arithmetic into Python

Rejected. Rust remains the production psychometric backend. Python remains parity/reference/fallback and orchestration only.

## Documentation and compatibility

- Add APA 7 doctoring with the squared-norm identity, byte-budget rationale, numerical round-off, environment-specific evidence, failure behavior, and rollback.
- Add an authoritative changelog fragment and render the managed `CHANGELOG.md` block before merge.
- No public Python signature, serialized fit result, database object, model identity, or Rust numerical contract changes.
- Rollback restores the former formulas but reintroduces the documented availability risk.

## References

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

NumPy Developers. (2026). *NumPy 2.5 manual: `numpy.matmul`*. https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

NumPy Developers. (2026). *NumPy 2.5 manual: `numpy.maximum`*. https://numpy.org/doc/stable/reference/generated/numpy.maximum.html

NumPy Developers. (2026). *NumPy 2.5 manual: `numpy.sqrt`*. https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
