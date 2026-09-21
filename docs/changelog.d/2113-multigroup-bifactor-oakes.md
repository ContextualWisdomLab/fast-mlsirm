# Joint multigroup bifactor Oakes ML information (#2113)

## Added

- Add `bifactor_multigroup_oakes_se` in Rust, PyO3, and Python for joint ML
  information and SEs: common item parameters enter once, free items enter by
  group, and focal general means/variances and optional specific variances
  enter jointly. Non-PD information keeps its matrix and reports unavailable
  SEs.
- Calibration command for s1 (pending): `python scripts/bifactor_multigroup_oakes_calibration.py --replicates 20 --persons-per-group 340 --q-general 121 --q-specific 121 --fd-step 1e-6 --seed 2113`.
- empirical SD / mean Oakes SE ratio: PENDING (merge gate)
