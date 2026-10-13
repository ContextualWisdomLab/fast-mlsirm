# Joint multigroup bifactor Oakes ML information (#2113)

## Added

- Add `bifactor_multigroup_oakes_se` in Rust, PyO3, and Python for joint ML
  information and SEs: common item parameters enter once, free items enter by
  group, and focal general means/variances and optional specific variances
  enter jointly. Non-PD information keeps its matrix and reports unavailable
  SEs.
- Synthetic SE calibration (s1, 3 groups x 340 persons, q=121, `a_general:0`,
  20 replicates pooled over seeds 2113/12113/22113/32113 at `--replicates 5`):
  empirical SD / mean Oakes SE = 1.107, paired bootstrap 95% interval
  [0.776, 1.349]. The interval covers 1 but cannot detect miscalibration below
  about 25%. Reproduce with `python scripts/bifactor_multigroup_oakes_calibration.py
  --replicates 5 --persons-per-group 340 --q-general 121 --q-specific 121
  --fd-step 1e-6 --seed <seed>`.
