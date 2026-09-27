Independent exact-head verification of 0bea6aeeeef1f8d0ba9bceab65fc34a8a03ca228 against current main 2caf37dae79e481108a0aa8344c8fc71c59a8955. Synthetic integration tree: 3728fb0df136a60bd19e81a349c6fe1f78af891b.

Sidecar contract suite: 31 passed locally and 31 passed with GITHUB_ACTIONS=true and warnings as errors. bash syntax and git diff --check passed.

Independent read-only probe on runner guest 02: the selected 3.12.14 executable reported 3.12.3 under the inherited library environment. With this exact patched startup prefix and the same executable, it reported 3.12.14 and passed logging/asyncio imports. The fast-mlsirm #2232 Noema job crashed with SIGSEGV/139 in logging before a review verdict; this patch addresses the shared runtime mismatch. This is not a claim that the hosted review passed or that immutable release-helper pins already consume this change.

Hosted checks remain queued; CodeRabbit and Devin statuses are successful. Maintainer-authorized exact-head bypass after local verification. Security and review workflows remain enabled.
