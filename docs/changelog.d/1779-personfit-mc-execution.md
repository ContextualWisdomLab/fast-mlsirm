# Execute deterministic person-fit Monte Carlo acceptance

## Fixed

Run the existing 500-replication person-fit U3 reversed-respondent acceptance in the normal Rust test suite instead of leaving it behind `#[ignore]`. The simulation design, respondent/item counts, fixed-seed reproducibility, public `person_fit_np` call, and >=95% detection criterion are unchanged.

The repository execution contract inspects the attributes attached to this exact Rust test independently of attribute order and rejects both Rust `ignore` syntaxes: the MetaWord form `#[ignore]` and the MetaNameValueStr reason form `#[ignore = "..."]`. Either form removes the test from ordinary execution, so neither may silently remove this scientific acceptance from the normal suite.

Authoritative syntax/behavior basis: *The Rust Reference*, “Testing attributes — The `ignore` attribute,” https://doc.rust-lang.org/reference/attributes/testing.html#the-ignore-attribute (accessed 2026-09-07).
