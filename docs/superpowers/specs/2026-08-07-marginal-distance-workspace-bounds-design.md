# Bounded Marginal Distance Workspaces Design

## Status

Approved for autonomous implementation under issue #563. This is a bounded reliability and performance slice for the NumPy reference/fallback MMLE path. It does not create a second production estimator and does not change the Rust-first product boundary.

## Problem

The marginal latent-space reference path currently controls the dominant person/item/quadrature working array but does not model every Euclidean-distance workspace. In particular, a broadcast subtraction of item positions and latent-space nodes can materialize an `n_items × n_x × latent_dim` array before reduction. Replacing the reduction with `numpy.einsum` removes only the subsequent square-product temporary; it does not bound or remove the broadcast workspace itself.

This creates a buyer-visible availability risk: otherwise valid dimensions can cause a large allocation before a stable package-owned error is raised. The NumPy reference/fallback path must remain numerically equivalent to the Rust estimator while failing closed before oversized workspaces are allocated.

## Selected approach

Use a shared pairwise-distance helper based on the squared-norm identity

\[
\lVert z_i-x_j\rVert^2
=
\lVert z_i\rVert^2+
\lVert x_j\rVert^2-
2z_i^\top x_j.
\]

The helper returns only the required `n_items × n_x` result and never materializes an `n_items × n_x × latent_dim` difference tensor. A separate checked-product guard validates actual temporary and output dimensions before allocation:

- item-local gradient workspace: `n_x × latent_dim`;
- item-position and node inputs: `n_items × latent_dim` and `n_x × latent_dim`;
- pairwise result workspace: `n_items × n_x`.

The existing 100,000,000-element marginal working-set ceiling remains the common resource ceiling unless repository evidence justifies a smaller public contract. Product behavior must not rely on integer overflow or an attempted allocation to enforce the limit.

## Alternatives rejected

### Keep the broadcast and add only `einsum`

Rejected. `einsum` can avoid an elementwise-square temporary, but it does not remove the preceding three-dimensional broadcast subtraction.

### Chunk every distance computation immediately

Deferred. Chunking can reduce peak memory further but adds loop and tuning complexity. The two-dimensional squared-norm formulation removes the largest unnecessary allocation first. Chunking remains a follow-up only if measured pairwise-output memory is still a practical bottleneck.

### Move new production arithmetic into Python

Rejected. Rust remains the production psychometric backend. Python is a parity/reference/fallback path and orchestration boundary.

## Interfaces

The implementation should add private helpers in `python/fast_mlsirm/estimators/marginal.py`:

```python
def _checked_marginal_workspace_product(
    label: str,
    *dimensions: int,
    limit: int = MAX_MARGINAL_WORKING_SET,
) -> int:
    """Return a bounded non-negative dimension product or fail closed."""


def _validate_marginal_distance_workspaces(
    *,
    n_items: int,
    n_x: int,
    latent_dim: int,
    uses_space: bool,
) -> None:
    """Validate every actual Euclidean-distance workspace before allocation."""


def _pairwise_euclidean_distances(
    left: np.ndarray,
    right: np.ndarray,
    *,
    eps_distance: float,
) -> np.ndarray:
    """Return bounded row-wise pairwise Euclidean distances without a 3-D tensor."""
```

Names may be refined only if all tests, documentation, and callers use one consistent contract.

## Data flow

1. Normalize and validate estimator dimensions.
2. Derive the exact latent-node count before constructing the node grid.
3. If the selected model uses latent space, validate distance input, gradient, and pairwise-output products.
4. Construct quadrature/node arrays.
5. Use the shared pairwise helper in table construction, the distance-weight update, and the covariate update.
6. Retain only the item-local `n_x × latent_dim` difference needed for the zeta gradient, guarded before the first M-step.
7. Preserve existing likelihood, gradient, stopping, missingness, multigroup, multilevel, zero-inflation, and scoring behavior.

## Error behavior

- Boolean, negative, non-integral, or otherwise malformed workspace dimensions fail with stable `ValueError` messages.
- Products above the configured element limit fail before NumPy allocates the relevant workspace.
- Finite moderate arrays produce finite distances; small negative round-off in the squared-distance identity is clamped to zero before adding `eps_distance`.
- Shape mismatch, non-finite `eps_distance`, or invalid array rank fails closed if the helper is called directly.
- Errors must not include response contents, source text, or provider-controlled values.

## Testing

TDD regressions must cover:

1. exact equivalence to the explicit broadcast formula on deterministic finite matrices;
2. zero-distance and near-round-off cases;
3. oversized pairwise output rejected from integer dimensions without allocating it;
4. oversized item-local gradient workspace rejected;
5. Boolean, negative, zero, and non-integer edge cases;
6. source-level absence of the three-dimensional `x_grid[None, :, :] - zeta[:, None, :]` pattern;
7. a realistic partially observed estimator case preserving outputs within tight floating-point tolerance;
8. complete added-production statement and branch coverage plus public docstrings where applicable;
9. exact-head Python, Rust/PyO3, package, explicit GPU-no-skip, fuzz, Security Scan, and SAST evidence.

## Performance evidence

Benchmark evidence must report dimensions, dtype, NumPy version, linked BLAS, operating system, processor, warm-up count, repetitions, elapsed distribution, and peak resident or traced Python allocation. It may claim removal of a specific broadcast temporary. It must not claim a universal speedup.

## Documentation and compatibility

- Add APA 7th doctoring with the squared-norm identity, resource-bound rationale, numerical round-off treatment, and rollback.
- Update an authoritative changelog fragment and render the managed `CHANGELOG.md` block before merge.
- No public Python signature, serialized fit result, database object, model identity, or Rust numerical contract changes.
- Rollback restores the former formulas but would reintroduce the documented availability risk.

## References

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

NumPy Developers. (2026). *NumPy 2.5 manual: `numpy.matmul`*. https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

NumPy Developers. (2026). *NumPy 2.5 manual: `numpy.einsum_path`*. https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html
