# Red main: support-policy version and a stale quadrature assertion

## Fixed

- `SECURITY.md` and `SUPPORT.md` still named `0.10.x` as the supported pre-1.0
  line after the 0.11 releases, so
  `tests/test_support_policy_version_contract.py` failed on `main` and blocked
  every PR's `python` check. Both now name `0.11.x`.
- `tests/test_bifactor_oakes.py::test_rejects_out_of_range_caller_arguments`
  still asserted that `q_general=5` is rejected. #1929 deliberately removed the
  Gauss-Hermite node-count cap — `SUPPORTED_Q` membership became a plain
  `q >= 1` check — and updated the same assertion in
  `tests/test_bifactor_grm.py` and `tests/test_bifactor_multigroup.py` but
  missed this file. The case now uses `q_general=0`, which is still invalid,
  and carries the same `#1929` note as its siblings.
