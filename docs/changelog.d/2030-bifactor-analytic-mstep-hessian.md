# Bifactor GRM analytic M-step Hessian (#2030)

## Changed

- `bifactor_grm::m_step_item` replaces the per-parameter finite-difference
  Hessian with a single-sweep analytic expected complete-data gradient and
  Hessian chained from `poly::grm_node_hessian` through the bifactor linear
  predictor (Bock & Aitkin, 1981, pp. 445, 448; Gibbons et al., 2007, eq. 9
  and Appendix A4–A6; same Term-A algebra as `bifactor_oakes::q_hessian_analytic`
  / Oakes, 1999, eq. 6). Line search still evaluates the objective only;
  production Newton no longer FD-reenters the quadrature grid.
- Optional M-step sweep counters (`enable_mstep_sweep_counters`) classify
  Base / FD / LineSearch / Newton for the #2022/#2030 recount contract.
- Unit tests: analytic-vs-FD item Hessian (block and general-only) and
  zero-FD Newton sweep accounting; ignored q=41/241 n=1020 recount for
  host evidence.
