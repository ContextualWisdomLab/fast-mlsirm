# Python 3.14 CI compatibility

## Changed

- Expanded the full Python CI job from a single CPython 3.12 runtime to a fail-slow CPython 3.12 and 3.14 matrix while preserving the existing Rust/PyO3 build, Rust-primary backend verification, package, GPU, fuzz, and security gates.
- Added a deterministic CI contract that requires Python 3.14 to execute the same complete pytest suite rather than a reduced compatibility smoke path.
- Added a non-matrix `python` aggregate job so branch-protection's required check context named `python` still receives a single SUCCESS/FAILURE after every matrix leg finishes.
