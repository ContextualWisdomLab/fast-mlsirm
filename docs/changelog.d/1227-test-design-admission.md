# Fixed-form test assembly admission safety

## Fixed

- Harden fixed-form assembly so form length and content-constraint controls are normalized before caller item evidence, complex/object information cannot be projected through `float64`, content labels are admitted as text without caller stringification, and exclusion indices must fit signed 64-bit item identity without narrowing overflow before the Rust-owned greedy assembly runs.
