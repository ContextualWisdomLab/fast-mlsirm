# Serving redundant parameter integrity

## Fixed

- Serving-bundle validation fails closed when exported redundant slope/distance-weight
  fields contradict canonical log-scale parameters, and admits only exact built-in
  numeric scalars so hostile conversion hooks cannot execute during load/score.
