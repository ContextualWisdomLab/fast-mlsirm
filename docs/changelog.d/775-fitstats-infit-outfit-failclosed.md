# Fit-statistics infit/outfit fail closed

## Fixed

- Public `infit_outfit()` requires the compiled `infit_outfit_stat` entrypoint and
  fails closed when the Rust core is missing or incomplete, matching S-X² and
  person-fit ownership after #771.
