# Release-source checkout authority

## Fixed

- The publication workflow no longer passes caller-controlled
  `workflow_dispatch` SHAs directly to `actions/checkout`. It first checks out
  the protected invocation commit (`github.sha`) with full history, requires the
  supplied control-plane identity to equal that commit, canonicalizes the
  requested release object, and proves that object is an ancestor of the trusted
  control plane.
- The verified release commit is emitted once from `verify-release`; all build,
  provenance, admission, tagging, and publication consumers use that job output.
  Invalid, unavailable, sibling, noncanonical, or mismatched identities fail
  closed before an untrusted tree becomes executable.
- Contract coverage executes the exact guard against real Git histories. The
  repair removes all 6 raw release checkout refs and both raw control-plane
  checkout refs that were present in the RED state.
