# S-X2 and person-fit require Rust ownership

## Fixed

- Public `s_x2()` and `person_fit()` fail closed when the compiled Rust core or required `s_x2_stat` / `person_fit_stat` entrypoints are missing, instead of silently selecting pure-Python numerical paths.
- Nonzero trait `prior_mean` for S-X² now routes through the existing native `s_x2_stat` prior vector instead of forcing the NumPy reference grid.
