# Response-time calibration semantic control safety

## Fixed

- Reject caller-defined numeric and truth-value protocols before response-time calibration controls are normalized or dispatched to the Rust core.
- Require the joint speed-accuracy Gauss-Hermite node count to be an exact supported integer instead of silently narrowing floating-point values.
- Keep required positive-finite runtime validation active under optimized Python execution instead of relying on `assert` guards that disappear with `-O`.
- Preserve positive-finite stopping, variance-floor, and fixed-speed-scale contracts while keeping all response-time likelihood and estimation arithmetic Rust-owned.
