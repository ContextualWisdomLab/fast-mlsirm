# Execute deterministic person-fit Monte Carlo acceptance

## Fixed

Run the existing 500-replication person-fit U3 reversed-respondent acceptance in the normal Rust test suite instead of leaving it behind `#[ignore]`. The simulation design, respondent/item counts, fixed-seed reproducibility, public `person_fit_np` call, and >=95% detection criterion are unchanged.

The repository execution contract now inspects the attributes attached to this exact Rust test independently of attribute order, so moving `#[ignore]` before or after `#[test]` cannot silently remove the scientific acceptance from the normal suite.
