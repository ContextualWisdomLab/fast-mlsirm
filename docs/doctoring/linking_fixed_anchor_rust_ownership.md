# Doctoring: Rust-owned fixed-anchor parameter linking

## Claim

Fixed common-item mean/mean-style scale and shift estimation, and the resulting
theta/alpha/b transformation onto the target metric, are owned by the compiled
Rust numeric core. Python validates public shapes and packages evidence.

## Standards and literature (APA 7th)

Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and linking:
Methods and practices* (3rd ed.). Springer.
https://doi.org/10.1007/978-1-4939-0317-7

## Verification

- Rust unit tests for identity and known scale/shift recovery.
- Python ownership sentinel requiring `core.link_fixed_item_parameters` transport.
