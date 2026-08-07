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

## Rejected implementations

A Boolean-to-float64 `astype` cannot reuse Boolean storage and therefore creates
a full N×J numeric mask. Likewise, `(resid2 / v)` creates another full N×J
result. A compound `(y - p) ** 2 * observed` expression also creates avoidable
full-size temporaries before the result is retained.

The accepted contract therefore requires:

- `np.subtract`, `np.square(..., out=...)`, and
  `np.multiply(..., out=...)` for one owned residual buffer;
- no numeric mask conversion;
- `np.sum(v, axis=0, where=observed)` for the masked variance reduction;
- `np.divide(resid2, v, out=resid2)` after retaining the infit numerator; and
- no change to public APIs, statistical definitions, clipping, missingness, or
  the Rust primary path.

## Verification

Permanent tests reconstruct the pre-change equations independently under sparse
missingness, an entirely missing item, and probabilities at both clipping
boundaries. Source-level allocation assertions prevent reintroduction of a
numeric mask, compound residual allocation, or separate division result.

The benchmark at `benchmarks/benchmark_fitstats_infit_outfit.py` records Python,
NumPy, linked numerical libraries, operating system, processor, dimensions,
dtype, missingness, warm-ups, repetitions, elapsed distributions, traced peak
memory, maximum RSS, and exact numerical difference. Results are
runtime-specific; no universal percentage improvement is claimed.

## Failure and rollback

Probability clipping keeps `v = p(1-p)` strictly positive. An item with no
observed responses retains the existing denominator floor and returns zero infit
and outfit. Any change to clipping, response validation, or missingness requires
rerunning parity and extreme-value contracts.

Rollback may restore the former equations but also restores the documented
full-size intermediates. An unverified alternate optimization is not a safe
fallback.

## MSA boundary

- Rust owns production fit-statistic arithmetic and optimized execution.
- Python owns validation, orchestration, and this parity fallback.
- This change adds no provider, credential, network, database, public API,
  dependency, model identity, or release-version boundary.

## References

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P.,
Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus,
M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F.,
Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with
NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2

NumPy Developers. (2026). *numpy.divide*. NumPy reference.
https://numpy.org/doc/stable/reference/generated/numpy.divide.html

NumPy Developers. (2026). *numpy.sum*. NumPy reference.
https://numpy.org/doc/stable/reference/generated/numpy.sum.html

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press.
