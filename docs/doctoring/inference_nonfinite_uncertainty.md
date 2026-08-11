# Non-finite covariance uncertainty semantics

## Decision

After observed-information inversion, diagonal elements of a covariance matrix may be non-finite when the numeric path encounters undefined or unbounded curvature. Mapping those diagonals to zero would report perfect certainty and is scientifically false. The Rust-owned `standard_errors_from_vcov` therefore:

1. returns `sqrt(d)` for finite positive `d`;
2. clamps finite non-positive `d` to `0.0` (numerical noise / negative curvature residue); and
3. preserves `NaN` and signed infinities unchanged.

Non-finite Hessian entries fail closed before inversion.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury.
