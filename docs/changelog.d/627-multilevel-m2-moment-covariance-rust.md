# Multilevel M2 moment and covariance ownership

- Move multigroup and multilevel M2 population-moment integration into the
  Rust/PyO3 numerical boundary, including the shared cluster-intercept
  reduction.
- Move the finite-cluster moment-covariance construction into Rust while
  preserving compact-label validation, finite-cluster correction, and the
  existing M2/RMSEA2 estimand.
- Keep the NumPy implementations available only as explicit parity references;
  public M2 paths fail closed when the required native entry point is absent.
