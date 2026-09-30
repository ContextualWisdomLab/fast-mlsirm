# Person-fit complete-data ZU3 reduction

## Changed

- Consolidated the complete-data ZU3 `s1`, `s2`, `s3`, `s4`, and beta-term moment reductions in the Rust `person_fit_np` owner into one item-order-preserving pass while retaining the existing public statistic and API.
- Added deterministic CPU `f64` production-output regression evidence requiring bitwise ZU3 parity with the pre-fold reduction order, including the established NaN semantics for perfect or degenerate response cases.
- This change claims reduced duplicate traversal and repeated arithmetic only; it does not claim a measured throughput multiplier, end-to-end latency improvement, or p95 result without repository-controlled benchmark evidence.
