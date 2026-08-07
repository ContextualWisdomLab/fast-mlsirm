# Bounded Marginal Distance Workspaces Design

## Status

Approved for autonomous implementation under issue #563. This is a bounded reliability and numerical-stability slice for the NumPy reference/fallback MMLE path. It does not create a second production estimator and does not change the Rust-first product boundary.

## Problem

The marginal latent-space reference path controls the dominant person/item/quadrature working array but did not model every Euclidean-distance allocation. In particular, a broadcast subtraction of item positions and latent-space nodes could materialize an `n_items × n_x × latent_dim` array before reduction. Replacing only the reduction with `numpy.einsum` would not bound or remove the broadcast workspace itself.

A second problem is unit ambiguity. The existing dominant gate is an **element-count** ceiling. Reusing that ceiling for a float64 distance matrix would authorize roughly 800 MB at 100,000,000 elements, so it would not solve the issue's hundreds-of-MiB availability risk. Distance workspaces therefore require a separate, explicit byte budget.

A third problem was exposed by RED numerical testing. The initially proposed bounded squared-norm identity

\[
\lVert z_i-x_j\rVert^2
=
\lVert z_i\rVert^2+
\lVert x_j\rVert^2-
2z_i^\top x_j
\]

can catastrophically cancel when `z_i` and `x_j` contain a large shared translation but their Euclidean separation is small. Algebraic equivalence is therefore insufficient for the fallback's floating-point contract.

## Selected approach

Use one shared pairwise-distance helper based on direct coordinate differences:

\[
\lVert z_i-x_j\rVert_2
=
\sqrt{\epsilon + \sum_k(z_{ik}-x_{jk})^2}.
\]

The helper returns only the required `n_items × n_x` float64 result and never materializes an `n_items × n_x × latent_dim` difference tensor. It keeps exactly one same-shaped reusable scratch matrix during the accumulation phase:

1. allocate the zeroed `L × R` distance accumulator;
2. allocate one `L × R` scratch matrix;
3. for each latent coordinate, subtract the two one-coordinate broadcast views into scratch with `np.subtract(..., out=scratch)`;
4. square scratch in place with `np.square(..., out=scratch)`;
5. add scratch into the accumulator;
6. release scratch;
7. add `eps_distance`; and
8. take the square root in place with `np.sqrt(..., out=distances)`.

This removes both the original three-dimensional tensor and the high-offset cancellation mechanism of the squared-norm/matmul formulation. It deliberately trades one same-sized two-dimensional scratch for stable coordinate subtraction. No universal speedup claim follows from that choice.

## Resource contract

Add a private named ceiling:

```python
MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024
```

The 128 MiB default is an initial conservative fallback-path limit, not a universal hardware capacity claim. It must be accompanied by representative compatibility and peak-memory evidence before merge. Larger workloads should use the Rust production backend rather than silently raising this fallback budget.

Use checked division before multiplication so malformed or astronomically large Python integers cannot create an unbounded intermediate product. Every active distance allocation is calculated in bytes using the actual float64 or boolean item size. The pairwise phases are modeled by lifetime rather than incorrectly added together:

- pairwise output: `L × R × 8` bytes;
- coordinate-subtraction kernel peak: output plus same-shaped float64 scratch, `2 × L × R × 8` bytes;
- final output-finiteness phase: output plus one `L × R` boolean mask;
- input-finiteness phases: one boolean mask for the corresponding already-owned left or right matrix;
- intentional item-local derivative workspace: `n_x × latent_dim × 8` bytes.

The dominant EM working-set ceiling remains a separate element-count contract and is checked through the same safe integer-product primitive. The two limits must not be conflated.

## Interfaces

Add private helpers in `python/fast_mlsirm/estimators/marginal.py`:

```python
def _checked_marginal_workspace_bytes(
    name: str,
    *dimensions: int,
    itemsize: int,
    limit_bytes: int,
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
    """Return bounded translation-stable float64 pairwise distances."""
```

`uses_space` is an exact Boolean. The direct helper accepts only exact base-class NumPy arrays that are two-dimensional, finite, float64, C-contiguous, and share one positive latent width. This prevents unreported dtype or layout conversion copies before the numerical kernel.

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
- Shape mismatch, non-finite matrices, non-float64 inputs, non-contiguous inputs, non-finite/non-positive epsilon, and invalid latent widths fail before numerical distance arithmetic.
- Moderate finite arrays produce finite distances. Exact and near-zero distances remain finite after positive `eps_distance` is added.
- Non-finite output after accumulation/square root fails closed.
- Errors contain no responses, source text, or provider-controlled values.

## Testing

RED→GREEN regressions cover:

1. tight equivalence to the explicit coordinate-difference equation;
2. deterministic large-common-offset translation stability for both `L < R` and `L > R`;
3. zero-distance and near-zero cases;
4. non-finite input rejection before numerical distance arithmetic;
5. exact float64 and boolean byte accounting by phase;
6. oversized pairwise and derivative workspaces rejected without allocation;
7. malformed dimensions, byte limits, item sizes, Boolean flags, layout, dtype, and epsilon;
8. checked failure for astronomically large Python integer dimensions;
9. an otherwise-valid public estimator request rejected before `_xi_nodes`;
10. `out=` reuse for subtract/square/sqrt and permanent absence of matmul/einsum and the 3-D distance broadcast from the governed helper;
11. realistic partially observed and covariate estimator parity at tight tolerances;
12. benchmark-schema tests that pin the coordinate-subtraction kernel name, exact output-plus-scratch peak, private ceiling, and safe reference comparison;
13. complete added-production statement/branch coverage and public docstrings; and
14. exact-head Python, Rust/PyO3, package, explicit GPU-no-skip, fuzz, Security Scan, and SAST evidence.

Source inspection is supplementary. Numerical and allocation evidence remains authoritative.

## Performance and compatibility evidence

Before Ready status, record:

- representative `n_items`, `n_x`, `latent_dim`, and missingness;
- dtype and contiguous-layout assumptions;
- Python, NumPy, linked BLAS, operating system, processor, and memory;
- warm-up and repetition counts;
- elapsed distribution;
- peak resident or traced Python allocation;
- parity against the direct coordinate-difference reference; and
- the largest repository test/fixture dimensions accepted by the 128 MiB ceiling.

The committed benchmark reports `pairwise_output_bytes` and the exact stable-kernel `pairwise_kernel_peak_bytes = 2 × pairwise_output_bytes`. The former 3-D broadcast is evaluated only below its independent safety ceiling. The change may claim removal of one specific 3-D broadcast, elimination of the reproduced high-offset cancellation mechanism, and enforcement of the private pairwise byte ceiling. It may not claim a universal speedup or universal memory suitability.

## Alternatives rejected

### Reuse the 100,000,000-element EM ceiling

Rejected. That value represents approximately 800 MB for float64 distance output and is not a byte-level availability contract.

### Keep the broadcast and add only `einsum`

Rejected. `einsum` can avoid an elementwise-square temporary but does not remove the three-dimensional subtraction.

### Squared-norm identity with one pairwise matrix

Rejected after RED testing. It bounds dimensionality but loses close Euclidean separations under sufficiently large common translations because `||z||² + ||x||² - 2zᵀx` subtracts large rounded quantities.

### Build several two-dimensional intermediates

Rejected. It lowers asymptotic dimensionality but can still multiply peak memory without improving the chosen stable arithmetic. The selected helper uses one output plus one reusable scratch.

### Chunk every distance computation immediately

Deferred. Chunking adds loop and tuning complexity. It remains a follow-up if the justified 128 MiB output-plus-scratch ceiling excludes required reference workloads.

### Move production arithmetic into Python

Rejected. Rust remains the production psychometric backend. Python remains parity/reference/fallback and orchestration only.

## Documentation and compatibility

- Add APA 7 doctoring with the direct coordinate-difference equation, byte-budget rationale, numerical cancellation evidence, environment-specific evidence, failure behavior, and rollback.
- Add an authoritative changelog fragment and render the managed `CHANGELOG.md` block before merge.
- No public Python signature, serialized fit result, database object, model identity, or Rust numerical contract changes.
- Rollback must not restore the squared-norm identity or original 3-D broadcast; a future replacement requires equivalent or stronger numerical and allocation evidence.

## References

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

Higham, N. J. (2002). *Accuracy and stability of numerical algorithms* (2nd ed.). Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898718027

NumPy Developers. (2026). *Broadcasting*. NumPy v2.5 manual. https://numpy.org/doc/stable/user/basics.broadcasting.html

NumPy Developers. (2026). *numpy.square*. NumPy v2.5 manual. https://numpy.org/doc/stable/reference/generated/numpy.square.html

NumPy Developers. (2026). *numpy.subtract*. NumPy v2.5 manual. https://numpy.org/doc/stable/reference/generated/numpy.subtract.html

NumPy Developers. (2026). *numpy.sqrt*. NumPy v2.5 manual. https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
