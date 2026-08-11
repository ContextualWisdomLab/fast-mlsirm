# Doctoring: Rust-owned fixed-form greedy assembly

## Claim

Public fixed-form maximum-information assembly with content min/max constraints is
owned by the compiled Rust numeric core. Python validates shapes and marshals
constraint maps; selection order and feasibility look-ahead do not re-implement the
greedy heuristic in production Python.

## Standards and literature (APA 7th)

van der Linden, W. J. (2005). *Linear models for optimal test design*. Springer.
https://doi.org/10.1007/0-387-29054-0

Lord, F. M. (1980). *Applications of item response theory to practical testing
problems*. Lawrence Erlbaum Associates.

## Verification

- Rust unit tests for unconstrained ranking, exclusion, content min/max, and
  infeasible constraint failure.
- Python ownership sentinel requiring `core.assemble_test_form_greedy` transport.
- Existing `tests/test_cov_a_test_design.py` behavioral suite.
