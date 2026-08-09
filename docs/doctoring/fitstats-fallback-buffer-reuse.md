# NumPy fit-statistics fallback buffer reuse

## Decision

The NumPy reference/fallback implementation of item infit and outfit reuses one
owned float64 squared-residual buffer and applies the Boolean observation mask in
place. It retains the masked residual column sum for infit, computes the
variance denominator with NumPy's `where=` reduction, and then reuses the
residual buffer for the outfit quotient.

The compiled Rust core remains the resolved production backend. This change is a
bounded parity and resource-safety correction in the existing Python fallback;
it does not establish a second production numerical authority.

## Previous allocation path

The former equations were mathematically valid:

\[
\operatorname{Outfit}_i
=
\frac{1}{n_i}
\sum_p
\frac{(y_{pi}-P_{pi})^2m_{pi}}{V_{pi}}m_{pi}
\]

\[
\operatorname{Infit}_i
=
\frac{
\sum_p (y_{pi}-P_{pi})^2m_{pi}
}{
\sum_p V_{pi}m_{pi}
}
\]

where \(m_{pi}\) is the Boolean observation indicator. A literal NumPy
translation, however, could create a full person-by-item quotient temporary and
a full numeric copy of the Boolean mask.

## Selected implementation

The fallback now performs the following bounded sequence:

1. allocate the owned float64 residual buffer with `np.subtract(y, p)`;
2. square it in place;
3. apply the Boolean mask in place;
4. retain its masked column sum for the infit numerator;
5. compute the infit denominator with `np.sum(v, axis=0, where=observed)`;
6. divide the same residual buffer by \(V\) in place; and
7. reduce that reused buffer for outfit.

This sequence preserves the previous probability clipping, missing-response
semantics, entirely missing item behavior, model identity, public API, and
result schema.

## Measurement boundary

Infit and outfit are descriptive item-fit summaries. They are not substitutes
for likelihood-based model fit, residual local-dependence diagnostics, parameter
recovery, uncertainty calibration, measurement invariance, or DIF analysis.
They should be interpreted with the model, sample size, item information,
missingness mechanism, and decision context.

An entirely missing item has a protected observation count of one and masked
numerators of zero, so both statistics remain zero rather than becoming
non-finite. This behavior is retained for compatibility and must not be
interpreted as evidence of good fit.

## Performance-claim boundary

The source contract proves removal of two named full-matrix allocation paths.
The included benchmark reports elapsed time and Python allocation observations
for one environment, matrix size, missingness pattern, NumPy/BLAS build, and
runtime. It does not prove universal memory reduction, universal speedup, or
production capacity. NumPy ufunc implementation details, allocator reuse,
hardware, and linked BLAS can alter observed results.

## Verification

Required same-head evidence includes:

- source-level absence of a numeric observation-mask copy and quotient
  expression;
- deterministic parity with the former equations under sparse missingness;
- an entirely missing item;
- both clipped-probability boundaries;
- observation that `np.divide` writes to the residual numerator buffer;
- observation that the variance reduction receives the original Boolean mask
  through `where=`;
- complete Python tests, statement/branch coverage, and public docstrings;
- Rust/PyO3, wheel reinstall, package acceptance, explicit GPU no-skip, fuzz,
  security, and SAST gates; and
- an unchanged-head independent review before protected merge.

## Rollback

Rollback restores the former mathematically equivalent expression but also
restores avoidable full-matrix allocation paths. If numerical parity fails,
prefer the previous expression temporarily while retaining the regression
fixtures and investigating the exact NumPy dtype, masking, or clipping boundary.
Do not silently change missingness or probability clipping to preserve a
performance result.

## APA 7 references

Wright, B. D., & Masters, G. N. (1982). *Rating scale analysis*. MESA Press.

Wright, B. D., & Stone, M. H. (1979). *Best test design*. MESA Press.

ISO/IEC. (2023). *ISO/IEC 25010:2023 systems and software engineering—Systems
and software quality requirements and evaluation (SQuaRE)—Product quality
model*. International Organization for Standardization.

NumPy Developers. (2026). *Universal functions (ufunc) and reductions*. NumPy
documentation.
