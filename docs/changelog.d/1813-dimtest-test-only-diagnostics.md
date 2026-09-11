# DIMTEST release-build diagnostic ownership

## Fixed

- Keep the per-group DIMTEST formula intermediates used by the independent Nandakumar & Stout oracle in test builds only, while release builds retain only the group contribution consumed by the production statistic. This removes the production dead-field warning without suppressing lints or changing DIMTEST arithmetic, public results, or the existing intermediate-value oracle.
