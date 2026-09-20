# Red `python` gate from lanes another job owns

## Fixed

- The fail-closed outcome gate (`tests/conftest.py`, Issue #1732) escalated an
  otherwise-green `python` matrix run to a failure on every pull request:
  6,977 tests passed and 8 skipped, and the eight were environmental rather
  than anything a PR could change. `python` is a required context, so this
  blocked merges repository-wide.
- Seven were GPU tests skipping because the `python` matrix has no Vulkan
  adapter, and one was `tests/test_fuzz_properties.py` failing to import
  because that matrix installs `requirements/ci.txt` without `atheris`.
- Both lanes are allowlisted with the job that actually owns their evidence.
  GPU parity is not waived: `gpu-smoke` installs a software Vulkan adapter,
  runs the explicit parity test, and fails the build when the JUnit report
  contains any skip. Coverage-guided fuzzing stays with the `fuzz` job, which
  installs the `.[fuzz]` extra and runs the Atheris harnesses directly.
