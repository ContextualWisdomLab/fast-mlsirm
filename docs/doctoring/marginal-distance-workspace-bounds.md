# Bounded marginal latent-distance workspaces

## Decision

The NumPy MMLE reference/fallback path computes Euclidean item-to-node
distances with the squared-norm identity

\[
\lVert \mathbf z_i-\mathbf x_j\rVert^2
=
\lVert\mathbf z_i\rVert^2
+
\lVert\mathbf x_j\rVert^2
-
2\mathbf z_i^\mathsf T\mathbf x_j.
\]

The implementation forms one `n_items × n_x` float64 matrix from
`left @ right.T` and mutates that output in place through scaling, row-norm
addition, round-off clamping, epsilon addition, and square root. It never
materializes the former `n_items × n_x × latent_dim` subtraction in the
covariate path.

This change applies only to the existing NumPy reference/fallback estimator.
Rust remains the production psychometric arithmetic boundary.

## Resource contract

Distance calculations use the private package-owned ceiling

```python
MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024
```

The 128 MiB value is an initial conservative fallback-path limit, not a claim
about universal hardware capacity. It is independent of the existing
100,000,000-element dominant EM working-set limit. Reusing that element limit
for float64 distances would authorize about 800 MB before row-norm or other
scratch allocations.

Before allocating latent nodes, the estimator validates:

- the intentional `n_x × latent_dim × 8` item-derivative workspace; and
- the pairwise output plus both live squared-row-norm vectors:
  `(n_items × n_x + n_items + n_x) × 8` bytes.

Dimensions, item size, byte limit, and the `uses_space` flag are checked without
Boolean coercion. Division-before-multiplication prevents astronomically large
Python integers from becoming an unbounded intermediate allocation shape.
Non-spatial MIRT bypasses the distance-specific gate while retaining the
separate dominant EM limit.

## Numerical contract

The pairwise helper accepts only finite, C-contiguous, two-dimensional float64
arrays with a shared positive latent width. It does not silently allocate a
dtype or layout conversion. `eps_distance` must be finite and strictly
positive.

Small negative values caused by floating-point cancellation are clamped to zero
before epsilon and square root are applied. A non-finite output remains a
failure rather than being serialized as a fitted result. The item-gradient path
retains its mathematically required `x_grid - zeta_i` workspace but reuses it in
place for the derivative instead of creating a second node-by-dimension matrix.

## Regression evidence

Permanent tests cover:

1. tight parity with the former explicit broadcast equation;
2. zero-distance and large-coordinate round-off behavior;
3. non-finite, malformed-shape, non-float64, non-contiguous, and invalid-epsilon
   rejection before BLAS;
4. exact byte accounting, including output-plus-row-norm peak-live scratch;
5. oversized pairwise and derivative failure before `_xi_nodes`;
6. Boolean, negative, fractional, zero, and astronomically large resource
   inputs;
7. source-level absence of the three-dimensional covariate subtraction;
8. deterministic partially observed repeated fits;
9. a valid covariate M-step delegating to the bounded helper without changing
   fitted evidence; and
10. the complete repository Python, Rust/PyO3, package, explicit GPU-no-skip,
    fuzz, security, and SAST gates on the final exact head.

Source inspection is supplementary. Numerical parity, resource-boundary tests,
and unchanged-head hosted evidence are the merge authority.

## Environment-specific compatibility evidence

The committed benchmark can be reproduced with:

```bash
python benchmarks/benchmark_marginal_distance_workspaces.py \
  --n-left 256 --n-right 512 --latent-dim 2 \
  --warmups 2 --repetitions 7
```

A development-container run on August 7, 2026 used CPython 3.13.5, NumPy 2.3.5,
OpenBLAS 0.3.30, Linux x86_64, five visible CPUs, and C-contiguous float64
inputs. For a 256 × 512 × 2 comparison:

- the governed pairwise output plus row norms required 1,054,720 modeled live
  bytes, below the 134,217,728-byte private ceiling;
- the bounded implementation had a median elapsed time of approximately
  0.000821 seconds and a median `tracemalloc` peak of 1,187,074 bytes;
- the legacy broadcast had a median elapsed time of approximately 0.003441
  seconds and a median `tracemalloc` peak of 5,244,072 bytes; and
- maximum absolute numerical difference was approximately
  \(1.65\times10^{-14}\).

These figures are environment-specific evidence only. `tracemalloc` does not
fully characterize native BLAS allocation, and process maximum RSS is
cumulative. The product claim is limited to removing the named three-dimensional
broadcast and enforcing the explicit byte ceiling; no universal speedup or
capacity claim is made.

The largest tensor Gauss-Hermite fixture allowed by the package remains bounded
by both the existing one-million-node tensor-grid check and the new distance
budget. Larger operational workloads belong on the Rust backend rather than
raising the Python fallback ceiling without new evidence.

## Failure and rollback

A resource violation raises a stable package-owned `ValueError` before latent
node or distance allocation. Diagnostics contain workspace labels and limits,
not responses or caller source text. Input finiteness and layout failures also
fail before matrix multiplication.

Rollback can restore the prior formulas without changing public signatures or
serialized results, but it reintroduces the documented three-dimensional
allocation risk. A safer future extension, if validated workloads exceed the
private output ceiling, is bounded row chunking with objective/gradient and
peak-memory parity—not an unreviewed ceiling increase.

## MSA boundary

- Rust owns production likelihood, gradients, optimization, multithreading,
  GPU execution, uncertainty, and recovery.
- Python owns this bounded reference/fallback implementation, validation,
  orchestration, and parity evidence.
- No LLM provider, credential, network dependency, database object, or public
  model identity is introduced.

## References

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P.,
Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus,
M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F.,
Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with
NumPy. *Nature, 585*(7825), 357–362.
https://doi.org/10.1038/s41586-020-2649-2

NumPy Developers. (2026). *numpy.matmul*. NumPy v2.5 manual.
https://numpy.org/doc/stable/reference/generated/numpy.matmul.html

NumPy Developers. (2026). *numpy.maximum*. NumPy v2.5 manual.
https://numpy.org/doc/stable/reference/generated/numpy.maximum.html

NumPy Developers. (2026). *numpy.sqrt*. NumPy v2.5 manual.
https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
