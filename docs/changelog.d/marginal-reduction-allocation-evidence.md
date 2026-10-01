# Marginal-reduction allocation evidence

## Changed

- Three allocation-reducing NumPy marginal-estimator expressions now route
  through private, directly testable helpers without changing their equations.
- `benchmarks/benchmark_marginal_reductions.py` compares those production
  helpers with the former broadcast equations on exactly representable inputs
  and emits environment, configuration, parity, elapsed observations, and
  Python-traced peak allocations as JSON. The report explicitly makes no
  universal speed or memory claim.
- Hand-derived unit fixtures and Rust/NumPy estimator-path tests protect the
  reduction axes, weights, and fitted-result parity.
