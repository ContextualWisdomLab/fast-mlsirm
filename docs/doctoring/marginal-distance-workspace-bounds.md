# Bounded marginal latent-distance workspaces

## Decision

The NumPy MMLE reference/fallback path computes Euclidean item-to-node distances
from coordinate differences before squaring:

\[
\lVert \mathbf z_i-\mathbf x_j\rVert_2
=
\sqrt{\epsilon + \sum_k(z_{ik}-x_{jk})^2}.
\]

The implementation allocates one `n_items × n_x` float64 output matrix and one
same-shaped reusable scratch matrix. For each latent coordinate it subtracts
into scratch with `np.subtract(..., out=scratch)`, squares that scratch in place,
and accumulates it into the output. Scratch is released before epsilon addition,
in-place square root, and output-finiteness validation. It never materializes an
`n_items × n_x × latent_dim` subtraction tensor.

The predecessor implementation used the algebraically equivalent squared-norm
identity

\[
\lVert \mathbf z_i-\mathbf x_j\rVert^2
=
\lVert\mathbf z_i\rVert^2
+
\lVert\mathbf x_j\rVert^2
-
2\mathbf z_i^\mathsf T\mathbf x_j.
\]

That identity can lose the small residual when both operands have a large common
offset because three large rounded terms are subtracted. A deterministic
high-offset regression now protects the direct Euclidean semantics for both
`n_left < n_right` and `n_left > n_right`. This is a numerical-stability fix,
not a change to the statistical model.

This change applies only to the existing NumPy reference/fallback estimator.
Rust remains the production psychometric arithmetic boundary.

## Resource contract

Distance calculations use the private package-owned ceiling

```python
MAX_MARGINAL_DISTANCE_WORKSPACE_BYTES = 128 * 1024 * 1024
```

The 128 MiB value is an initial conservative fallback-path limit, not a claim
about universal hardware capacity. It is independent of the existing
100,000,000-element dominant EM working-set limit.

Before allocating latent nodes, the estimator validates the intentional
`n_x × latent_dim × 8` item-derivative workspace and the pairwise kernel phases.
For an `L × R` pairwise result, the dominant stable-kernel phase is two
float64 pairwise matrices live at once (output plus reusable scratch), or
`2 × L × R × 8` bytes. Input-finiteness validation and final-output validation
are modeled separately with their boolean masks; those phases are checked
against the same ceiling rather than incorrectly summing non-overlapping
lifetimes.

Dimensions, item size, byte limit, and the `uses_space` flag are checked without
Boolean coercion. Division-before-multiplication prevents astronomically large
Python integers from becoming an unbounded intermediate allocation shape.
Non-spatial MIRT bypasses the distance-specific gate while retaining the
separate dominant EM limit.

## Numerical contract

The pairwise helper accepts only finite, exact base-class NumPy arrays that are
C-contiguous, two-dimensional, float64, and share one positive latent width. It
does not silently allocate a dtype or layout conversion. `eps_distance` must be
finite and strictly positive.

Coordinate subtraction occurs before squaring, so a large common translation
does not create the cancellation mechanism inherent in the predecessor
squared-norm identity. `np.subtract`, `np.square`, and `np.sqrt` all receive
explicit `out=` buffers. NumPy 2.5 documents that a ufunc result is stored into
the supplied `out` array when one is provided, and its broadcasting guidance
notes that broadcasting can be memory-inefficient in some cases; the kernel
therefore broadcasts only two one-coordinate views into the already allocated
two-dimensional scratch rather than materializing a three-dimensional result.
A non-finite output remains a failure rather than being serialized as a fitted
result.

The item-gradient path retains its mathematically required
`x_grid - zeta_i[None, :]` node-by-latent-dimension workspace. That bounded
workspace is distinct from the pairwise helper and is preflighted before node
construction.

## Regression evidence

Permanent tests cover:

1. tight parity with the direct explicit coordinate-difference equation;
2. deterministic high-offset translation stability in both rectangular
   orientations;
3. zero-distance and near-zero behavior;
4. non-finite, malformed-shape, non-float64, non-contiguous, and invalid-epsilon
   rejection before the numerical kernel;
5. exact byte accounting, including output-plus-scratch and mask phases;
6. oversized pairwise and derivative failure before `_xi_nodes`;
7. Boolean, negative, fractional, zero, and astronomically large resource
   inputs;
8. source-level absence of matmul/einsum and the three-dimensional covariate
   subtraction from the governed helper;
9. deterministic partially observed repeated fits;
10. a valid covariate M-step delegating to the bounded helper without changing
    fitted evidence; and
11. the complete repository Python, Rust/PyO3, package, explicit GPU-no-skip,
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

A predecessor-head development-container run on August 7, 2026 used CPython
3.13.5, NumPy 2.3.5, OpenBLAS 0.3.30, Linux x86_64, five visible CPUs, and
C-contiguous float64 inputs. It measured the superseded squared-norm/matmul
kernel, so its latency and memory numbers are retained only as historical
engineering evidence and are **not** acceptance evidence for the stable kernel.
The final coordinate-subtraction implementation must be benchmarked again on
its exact unchanged head before Ready status; no predecessor measurement
transfers across the head change.

The largest tensor Gauss-Hermite fixture allowed by the package remains bounded
by both the existing one-million-node tensor-grid check and the new distance
budget. Larger operational workloads belong on the Rust backend rather than
raising the Python fallback ceiling without new evidence.

## Failure and rollback

A resource violation raises a stable package-owned `ValueError` before latent
node or distance allocation. Diagnostics contain workspace labels and limits,
not responses or caller source text. Input finiteness and layout failures also
fail before arithmetic.

Rollback to the squared-norm identity is not acceptable because it would
restore the reproduced high-offset cancellation bug. A future extension, if
validated workloads exceed the private output ceiling, should use bounded row
or column chunking with objective/gradient, numerical-stability, and peak-memory
parity—not an unreviewed ceiling increase.

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

Higham, N. J. (2002). *Accuracy and stability of numerical algorithms* (2nd
ed.). Society for Industrial and Applied Mathematics.
https://doi.org/10.1137/1.9780898718027

NumPy Developers. (2026). *Broadcasting*. NumPy v2.5 manual.
https://numpy.org/doc/stable/user/basics.broadcasting.html

NumPy Developers. (2026). *numpy.square*. NumPy v2.5 manual.
https://numpy.org/doc/stable/reference/generated/numpy.square.html

NumPy Developers. (2026). *numpy.subtract*. NumPy v2.5 manual.
https://numpy.org/doc/stable/reference/generated/numpy.subtract.html

NumPy Developers. (2026). *numpy.sqrt*. NumPy v2.5 manual.
https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html

## References

Jeon, M., Jin, I. H., Schweinberger, M., & Baughman, M. (2021). Mapping unobserved item–respondent interactions: A latent space item response model with application to the International Personality Item Pool. *Psychometrika, 86*(2), 378–403. https://doi.org/10.1007/s11336-021-09762-5

Higham, N. J. (2002). *Accuracy and stability of numerical algorithms* (2nd ed.). Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9780898718027
