# Non-finite covariance uncertainty semantics

## Decision

After observed-information inversion, diagonal elements of a covariance matrix may be non-finite when the numeric path encounters undefined or unbounded curvature. Mapping those diagonals to zero would report perfect certainty and is scientifically false. The Rust-owned `standard_errors_from_vcov` therefore:

1. returns `sqrt(d)` for finite positive `d`;
2. clamps finite non-positive `d` to `0.0` only as the repository's existing compatibility treatment for negative roundoff/curvature residue that cannot produce a real-valued standard error; and
3. preserves `NaN`, positive infinity, and negative infinity unchanged.

Non-finite Hessian entries fail closed before inversion. Preserving non-finite sentinels is an explicit repository safety and representation decision; the measurement/statistical sources below motivate truthful uncertainty reporting and ordinary finite-positive standard-error semantics, while Rust's primary `f64` documentation establishes the implementation-level availability of distinct `NaN` and signed-infinity values.

## Source-specific rationale

- **AERA/APA/NCME Standards.** The Standards govern technically sound score interpretation and appropriate communication of measurement limitations. They support the no-false-precision invariant: undefined or unbounded uncertainty must remain observable rather than being rendered as exact zero uncertainty.
- **NIST/SEMATECH uncertainty guidance.** NIST defines standard uncertainty through root-sum-of-squares combination of standard-deviation components and expanded uncertainty through a coverage multiplier. This supports square-root-based finite-positive uncertainty semantics and the requirement to distinguish meaningful finite uncertainty from an undefined or unbounded state.
- **Rust `f64` primary documentation.** Rust exposes distinct `NAN`, `INFINITY`, and `NEG_INFINITY` constants and classification operations such as `is_nan` and `is_infinite`. The implementation therefore tests and preserves both infinity signs rather than collapsing them into one finite sentinel.
- **Statistical inference reference.** Covariance-derived standard errors use the square root of a meaningful finite variance. An undefined or unbounded covariance diagonal cannot truthfully be interpreted as zero standard error.

Changing the current finite-negative-diagonal compatibility rule from clamping to fail-closed behavior would be a separate scientific/API decision and requires its own test-first review; this document does not silently change that contract.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association. https://www.aera.net/Publications/Books/Standards-for-Educational-Psychological-Testing-2014-Edition

Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury.

National Institute of Standards and Technology. (n.d.). *Standard and expanded uncertainties*. In *NIST/SEMATECH e-Handbook of statistical methods*. https://www.itl.nist.gov/div898/handbook/mpc/section5/mpc57.htm

The Rust Project Developers. (2026). *Primitive type `f64`*. Rust documentation. https://doc.rust-lang.org/stable/core/primitive.f64.html
