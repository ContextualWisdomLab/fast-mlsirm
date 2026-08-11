# Fit-statistics S-X² and person-fit fail closed

## Fixed

- Public `s_x2()` and `person_fit()` fail closed when the compiled Rust core or
  required entrypoints are missing, and nonzero trait priors dispatch through
  the existing Rust `s_x2_stat` path instead of the Python ICC-grid reference.
