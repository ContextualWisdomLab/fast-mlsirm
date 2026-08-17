# G-theory NumPy scalar trust hardening

## Fixed

- Require exact package-supported NumPy integer and floating scalar classes for G-theory public numeric controls, rejecting caller-defined subclasses even when they spoof NumPy module metadata before any conversion callback can execute.
