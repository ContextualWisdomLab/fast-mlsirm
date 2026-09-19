# Red `python` gate from capability lanes another job owns

## Fixed

- The fail-closed outcome gate (`tests/conftest.py`, Issue #1732) escalated
  capability-gated skips in the ordinary `python` matrix. The first repair
  over-broadly allowed the whole high-q module and incorrectly described the
  Atheris harnesses as evidence for a separate Hypothesis module.
- The allowlist now names six exact high-q pytest nodes instead of a module
  glob. `gpu-smoke` installs a software Vulkan adapter, sets
  `STAGE5_HIGH_Q=1`, executes those six nodes plus the existing marginal GPU
  parity node, and fails when its JUnit evidence contains any skip.
- `tests/test_fuzz_properties.py` is no longer allowlisted. The `fuzz` job
  installs the `.[fuzz]` dependencies, executes that Hypothesis module
  directly, and rejects collection-time or runtime skips before running the
  separately owned Atheris harnesses.
- `test_allowlisted_capability_nodes_have_exact_ci_owners` prevents a future
  module glob, owner-name substitution, or allowlisted node without an
  executable CI command.
