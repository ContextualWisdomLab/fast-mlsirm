# In-place NumPy fallback reductions for item infit and outfit

## Decision

The compiled Rust core remains the production implementation for item infit and
outfit. The NumPy body is a tested reference/fallback path. Its equations remain

\[
\operatorname{Outfit}_j
=
\frac{1}{n_j}
\sum_p
\frac{(y_{pj}-p_{pj})^2 I_{pj}}
     {p_{pj}(1-p_{pj})},
\]

\[
\operatorname{Infit}_j
=
\frac{
\sum_p (y_{pj}-p_{pj})^2 I_{pj}
}{
\sum_p p_{pj}(1-p_{pj})I_{pj}
}.
\]

Here `I` is the Boolean observation mask. The fallback stores the masked squared
residual in one float64 work buffer, retains its column sum for the infit
numerator, divides that same buffer in place for outfit, and computes the infit
denominator with NumPy's `where=` reduction. This avoids a numeric N×J mask copy
and a separate N×J division result.

## Rejected initial implementation

The first proposed optimization cast the Boolean mask with
`observed.astype(v.dtype, copy=False)` and passed the float mask to `einsum`.
A Boolean-to-float64 conversion cannot reuse the Boolean storage, so that path
allocated a new N×J array and did not support the claimed allocation reduction.
It also retained `(resid2 / v)`, which creates another N×J result.

A later revision removed those two allocations but still constructed the masked
squared residual through a compound expression and relied on a loose
process-level peak-memory assertion rather than a numerical oracle. The accepted
contract therefore requires:

- `np.subtract`, `np.square(..., out=...)`, and
  `np.multiply(..., out=...)` for one owned residual buffer;
- no `observed.astype(...)` numeric mask;
- `np.sum(v, axis=0, where=observed)` for the masked variance reduction;
- `np.divide(resid2, v, out=resid2)` after retaining the infit numerator; and
- no change to public APIs, statistical definitions, clipping, missingness, or
  the Rust primary path.

## Verification

Permanent tests reconstruct the pre-change equations independently under
realistic sparse missingness, an entirely missing item, and probabilities at
both clipping boundaries. Source-level allocation assertions supplement the
numerical tests by preventing reintroduction of a float mask cast, compound
residual allocation, or separate division result.

The benchmark at `benchmarks/benchmark_fitstats_infit_outfit.py` records Python,
NumPy, linked numerical libraries, operating system, processor, dimensions,
dtype, missingness, warm-ups, repetitions, elapsed distributions, traced peak
memory, maximum RSS, and exact numerical difference. Results are
runtime-specific; no universal percentage improvement is claimed.

## Failure and rollback

Probability clipping keeps `v = p(1-p)` strictly positive, so in-place division
remains finite for validated dichotomous inputs. An item with no observed
responses retains the repository's existing denominator floor and returns zero
infit and outfit. Any future change to clipping, response validation, or
missingness requires rerunning the parity and extreme-value contracts.

Rollback may restore the former equations without changing serialized results,
but it also restores the documented full-size intermediate allocations. The
preferred fallback for any regression is the pre-change expression rather than
an unverified alternative optimization.

## MSA and product boundary

- Rust owns production fit-statistic arithmetic and optimized execution.
- Python owns validation, orchestration, and this reference/fallback parity
  implementation.
- The change adds no provider, credential, network, database, model identity,
  public API, dependency, or release-version boundary.

## References

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P.,
Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus,
M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F.,
Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with
NumPy. *Nature, 585*(7825), 357–362.
https://doi.org/10.1038/s41586-020-2649-2

NumPy Developers. (2026). *numpy.divide*. NumPy reference.
https://numpy.org/doc/stable/reference/generated/numpy.divide.html

NumPy Developers. (2026). *numpy.sum*. NumPy reference.
https://numpy.org/doc/stable/reference/generated/numpy.sum.html

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press.
