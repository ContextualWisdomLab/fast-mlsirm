# Mokken integer-control overflow

## Fixed

- `mokken_analysis` no longer leaks Python's `OverflowError` when an exact
  built-in `int` `lower_bound` or `alpha` lies beyond binary64 range (for
  example `10**400`). Such controls now fail the package semantic-control
  contract with `ValueError("<name> must be finite")` before response traversal
  or compiled-core discovery (#1671). NumPy integer scalars and 0-D integer
  arrays are bounded by 64-bit storage and already convert finitely.
