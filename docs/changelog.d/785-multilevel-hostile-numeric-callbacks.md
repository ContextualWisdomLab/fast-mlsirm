# Multilevel hostile numeric callback rejection

## Fixed

- Multilevel membership weights and AR(1) coefficients now admit only exact
  built-in `int`/`float` scalars, rejecting booleans and caller-defined
  conversion hooks before contract arithmetic.
