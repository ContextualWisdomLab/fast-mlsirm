# Residual-map explained share

## Added

- The Rust-owned residual interaction-map result now returns the cellwise
  squared reconstruction share alongside reconstruction, unexplained residual,
  and cross-share evidence. Python consumers receive the value as a projection
  of the native result and do not recalculate the psychometric arithmetic.
