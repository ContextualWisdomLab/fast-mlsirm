# Non-finite covariance uncertainty semantics

## Decision

After observed-information inversion, diagonal elements of a covariance matrix may be non-finite when the numeric path encounters undefined or unbounded curvature. Mapping those diagonals to zero would report perfect certainty and is scientifically false. The Rust-owned `standard_errors_from_vcov` therefore:

1. returns `sqrt(d)` for finite positive `d`;
2. clamps finite non-positive `d` to `0.0` (numerical noise / negative curvature residue); and
3. preserves `NaN` and signed infinities unchanged.

Non-finite Hessian entries fail closed before inversion.

## References

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association. [Official AERA edition and open-access files](https://www.aera.net/Publications/Books/Standards-for-Educational-Psychological-Testing-2014-Edition). The Standards require technically sound, appropriately documented score interpretations; preserving undefined or unbounded uncertainty makes that limitation observable instead of presenting a false zero-uncertainty claim.

Casella, G., & Berger, R. L. (2002). *Statistical inference* (2nd ed.). Duxbury. [WorldCat bibliographic record](https://search.worldcat.org/title/Statistical-inference/oclc/67327073). The text supplies the mathematical basis for covariance-derived standard errors: finite positive variance gives a square-root standard error, while an undefined or unbounded variance cannot be treated as zero. The implementation therefore clamps only finite numerical residue and preserves `NaN` and signed infinities.
