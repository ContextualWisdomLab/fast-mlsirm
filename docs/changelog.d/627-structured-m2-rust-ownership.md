# Structured M2 Rust ownership

- Route public single-population `m2()` calls that include estimated population
  moments, anchored items, or a fixed spatial coefficient through the Rust/PyO3
  M2 kernel. Missing structured native capability now fails closed instead of
  entering the NumPy reference implementation.
- Preserve the existing M2 estimand and degrees-of-freedom contract while
  moving finite-difference calibration and population nuisance columns into
  the Rust numerical owner.
