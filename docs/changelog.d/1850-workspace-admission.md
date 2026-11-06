# Explicit polytomous LSIRM workspace admission

## Added

Polytomous LSIRM callers can supply `workspace_budget_bytes` to reject an
estimated core workspace exceeding their own resource policy before grid and
table allocation. The Rust API retains `fit_poly_lsirm` and adds
`fit_poly_lsirm_with_budget`; Python accepts an optional positive integer.
The existing numerical model and supported quadrature rules are unchanged.

The conservative estimate covers logical Rust vector payloads, nested row
headers, numerical scratch space, score outputs, and convergence receipts.
It excludes caller inputs, Python conversion buffers, allocator overhead and
retained pages; it is not a process RSS limit. An omitted budget preserves the
existing policy and does not prevent memory exhaustion. Hosted consumers must
choose an explicit budget and retain their process isolation/resource controls.
