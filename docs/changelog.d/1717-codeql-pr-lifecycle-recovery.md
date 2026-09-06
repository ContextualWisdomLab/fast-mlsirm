# CodeQL PR lifecycle recovery

## Fixed

- Make repository-owned CodeQL participate in the pull-request lifecycle used by CI and ClusterFuzzLite: `opened`, `synchronize`, `reopened`, `ready_for_review`, `converted_to_draft`, and `closed`.
- Prevent Draft/closed events from running expensive analysis while allowing an unchanged source head to reacquire the required `Analyze (actions)` context when the PR becomes Ready.
- Preserve `workflow_dispatch`, the explicit Ubuntu 24.04 runner, exact-PR concurrency, and the protected `Analyze (actions)` context.
- Bind every acceptance-step artifact admitted to the buyer evidence packet to the acceptance summary's resolved evidence root. Artifact paths that resolve outside that root, including symlink escapes, now fail closed instead of being flattened to a basename and silently admitted.
- Keep benchmark and release-index HTML evidence inside each manifest's resolved evidence root and verify the recorded HTML SHA-256 before adding it to the buyer packet.
- Bind the buyer packet's `source_commit` to the release-acceptance summary's sealed `source_commit`; acceptance evidence from another revision now fails closed instead of being relabeled as current evidence.
- Require both release acceptance and sales readiness to report `status: ok` before their evidence can be promoted into a buyer packet whose own status is `ok`.

Source-level RED `552e753fbbc83a46763f8d65bcc4b56912a986b5` requires the complete CodeQL lifecycle and inactive-PR suppression. Causal GREEN `b1e7c9f4fcd8d7e50b24bc73da16726082272f2b` implements that contract. PR #1742 reproduced the deadlock on unchanged head `dbb6a9bf74e940280fc5b0c247469b7850534709`: its Ready transition created successful CI run `33935708280`, while the earlier CodeQL run `33925007681` stayed cancelled and no Ready-event CodeQL run materialized.

Buyer-packet provenance RED `352d906229dc84b00b8ef74352c5acd25a753313` requires `_collect_files` to reject an acceptance artifact outside `acceptance_path.parent.resolve()`. Causal GREEN `837b920fb83b1d49f29b4c7b4af34e691d52eb3d` removes the outside-root basename fallback, preserves exact root-relative archive names, and rejects resolved path escape before packet construction.

Linked-report provenance RED `75a68680a5fb0a749ca718d222a6c11e17f017bb` and causal GREEN `f266cd7521d038813bba2e7465664586d5503f63` confine benchmark/release HTML to their manifest roots. Follow-up digest verification at `6ea912c99de2a9ad59cbd9cad5b7895fb1066b1f` rejects modified HTML when its bytes no longer match the manifest-recorded SHA-256.

Cross-revision acceptance RED `b2c3a792aadc7cce08b0e57db364fbdfdfd2ff8f` proves that a buyer packet could previously accept a valid-looking acceptance summary from a different source revision and then stamp the packet with the current repository `HEAD`. Causal GREEN `67cbf6d514c446f1826a8aefed23e0ee2cb788aa` captures the packet source identity once, requires the acceptance summary to carry that exact sealed commit, and reuses the same identity in the emitted manifest. Fixture alignment at `a5a4950dfe105b8ac64467ed0981b77b5b471c15` makes existing positive packet tests carry the same sealed source identity as production acceptance.

Upstream-status RED `087588114ef4bd0aefdf56465cc937147d51ac28` proves that failed release-acceptance or sales-readiness manifests could previously be embedded while the buyer packet itself still emitted `status: ok`. Causal GREEN `6ae7fdfa412d5619364cc26298e2de22bf6d4160` reads both bounded manifests before collection and fails closed unless each reports `status: ok`.
