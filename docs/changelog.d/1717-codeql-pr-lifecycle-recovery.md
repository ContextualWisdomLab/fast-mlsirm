# CodeQL PR lifecycle recovery

Repository-owned CodeQL now participates in the same pull-request lifecycle used by CI and ClusterFuzzLite: `opened`, `synchronize`, `reopened`, `ready_for_review`, `converted_to_draft`, and `closed`.

The prior bare `pull_request:` trigger used GitHub's default activity set. After a Draft PR was marked Ready on an unchanged source head, CI could reacquire executable exact-head evidence while the required `Analyze (actions)` CodeQL context remained cancelled or absent. That made policy-compliant recovery depend on unrelated source churn.

Source-level RED `552e753fbbc83a46763f8d65bcc4b56912a986b5` adds an executable contract requiring queue-sensitive CodeQL to receive the complete lifecycle and to suppress expensive analysis for Draft/closed events. Causal GREEN `b1e7c9f4fcd8d7e50b24bc73da16726082272f2b` adds the lifecycle types and an active-PR guard to `analyze-actions` while preserving `workflow_dispatch`, the explicit Ubuntu 24.04 runner, exact-PR concurrency, and the protected `Analyze (actions)` context.

Observed reproduction: PR #1742 stayed on exact head `dbb6a9bf74e940280fc5b0c247469b7850534709`; its Ready transition produced CI run `33935708280`, which completed successfully, while the earlier CodeQL run `33925007681` remained cancelled and no Ready-event CodeQL run materialized. This repair removes that lifecycle deadlock without a no-op commit, force update, gate weakening, or predecessor-evidence transfer.
