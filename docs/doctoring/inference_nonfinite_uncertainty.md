# Non-finite covariance uncertainty semantics

## Decision

After observed-information inversion, diagonal elements of a covariance matrix may be non-finite when the numeric path encounters undefined or unbounded curvature. Mapping those diagonals to zero would report perfect certainty and is scientifically false. The Rust-owned `standard_errors_from_vcov` therefore:

1. returns `sqrt(d)` for finite positive `d`;
2. clamps finite non-positive `d` to `0.0` (negative roundoff/curvature residue on a path that cannot yield a real-valued standard error); and
3. preserves `NaN` and signed infinities unchanged.

Non-finite Hessian entries fail closed before inversion. This representation rule is an explicit repository safety contract, not a claim that the cited measurement/statistical sources prescribe a particular IEEE-754 sentinel policy.

## Source-specific rationale

- **AERA/APA/NCME Standards.** The Standards require uncertainty, precision, and score-interpretation evidence to be represented in ways that support defensible interpretations. They motivate the no-false-precision invariant: an undefined or unbounded uncertainty state must not be silently reported as exact zero uncertainty. The official joint-publisher site provides the 2014 edition as open access.
- **NIST/SEMATECH uncertainty guidance.** NIST defines standard uncertainty through root-sum-of-squares combination of standard-deviation components and ties expanded uncertainty to interval coverage. This grounds the ordinary finite-positive `sqrt(variance)` interpretation and the need to preserve the distinction between a meaningful zero and a non-finite uncertainty state.
- **Rust `f64` primary documentation.** Rust exposes distinct `NaN`, `INFINITY`, and `NEG_INFINITY` values and explicit `is_nan`/`is_infinite` classification. This is the implementation-level basis for preserving both infinity signs and testing them symmetrically rather than collapsing them to a finite sentinel.

Finite negative covariance diagonals do not define real standard errors. The existing compatibility rule clamps those finite non-positive values to zero; changing that interpretation to a hard failure would be a separate scientific/API decision requiring its own test-first review.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association. https://www.testingstandards.net/open-access-files.html

National Institute of Standards and Technology. (n.d.). *Standard and expanded uncertainties*. In *NIST/SEMATECH e-Handbook of statistical methods*. https://www.itl.nist.gov/div898/handbook/mpc/section5/mpc57.htm

The Rust Project Developers. (2026). *Primitive type f64*. Rust documentation. https://doc.rust-lang.org/stable/core/primitive.f64.html
