# CodeQL PR lifecycle recovery

## Fixed

- Make repository-owned CodeQL participate in the pull-request lifecycle used by CI and ClusterFuzzLite: `opened`, `synchronize`, `reopened`, `ready_for_review`, `converted_to_draft`, and `closed`.
- Prevent Draft/closed events from running expensive analysis while allowing an unchanged source head to reacquire the required `Analyze (actions)` context when the PR becomes Ready.
- Preserve `workflow_dispatch`, the explicit Ubuntu 24.04 runner, exact-PR concurrency, and the protected `Analyze (actions)` context.

Source-level RED `552e753fbbc83a46763f8d65bcc4b56912a986b5` requires the complete lifecycle and inactive-PR suppression. Causal GREEN `b1e7c9f4fcd8d7e50b24bc73da16726082272f2b` implements that contract. PR #1742 reproduced the deadlock on unchanged head `dbb6a9bf74e940280fc5b0c247469b7850534709`: its Ready transition created successful CI run `33935708280`, while the earlier CodeQL run `33925007681` stayed cancelled and no Ready-event CodeQL run materialized.
