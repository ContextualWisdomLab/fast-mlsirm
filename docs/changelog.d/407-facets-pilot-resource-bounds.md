# Facets-pilot resource bounds

## Fixed

- Reject persons-by-items-by-raters dense pilot designs above 1,000,000 cells
  before allocating Python tuple tensors or NumPy arrays, preventing bounded
  sparse observations from amplifying into memory-exhaustion workloads.
- Validate observed respondent, item, and rater support with precomputed sets
  before dense construction, replacing repeated full scans with linear-time
  membership checks.
