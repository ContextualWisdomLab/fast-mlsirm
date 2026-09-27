# Linux floor and NumPy runtime contract

## Changed

- `docs/release_acceptance.md` documents that fast-mlsirm Linux wheels target
  `manylinux2014` (glibc 2.17) while NumPy, an unbundled runtime dependency,
  keeps its own floor (NumPy 2.5: `manylinux_2_27`/`2_28`), and that research
  consumption must use a Fortran-free NumPy build.
