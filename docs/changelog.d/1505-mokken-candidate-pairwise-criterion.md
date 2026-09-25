# Enforce AISP pairwise Criterion 1

## Fixed

- Require every free AISP candidate to pass the current Bonferroni-adjusted, one-sided pairwise `Zij >= Z_c` test against every already-selected item before its candidate-level `Hi`/`Zi` and augmented-set `H` can admit it. This restores the published Mokken Criterion 1 (`Hij > 0` for every item pair) during Step 2 instead of allowing a zero or nonsignificant pair to be masked by a sufficiently strong candidate-level aggregate statistic. The start-pair significance screen is one-sided for the same reason. The CRAN `search.normal.R` negative-`Hij` prefilter is retained only as implementation evidence for candidate enumeration and Bonferroni counting where it is weaker than the peer-reviewed criterion; Koopman, Zijlstra, and van der Ark (2022, *Quality of Life Research*, 31, 25–36, https://doi.org/10.1007/s11136-021-02840-2) is the scientific authority for the pairwise acceptance invariant.
