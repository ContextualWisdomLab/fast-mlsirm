# CI queue and review-governance hardening

## Changed

- Pull-request CI runs now share a PR-number-scoped concurrency group, so a newer head cancels superseded queued or running CI evidence instead of consuming capacity for an obsolete commit.
- Push CI remains isolated by branch or ref and does not collide with pull-request validation.
- Draft pull requests no longer consume automatic CodeRabbit reviews, and automatic incremental review-on-every-push is disabled; maintainers request a final review only after a stable head is ready.
- The hourly read-only PR-governance workflow verifies its repository contract with the Python standard library, fails closed when no matching test is discovered, and no longer assumes that `pytest` is preinstalled on a fresh scheduled runner.

No test, security, packaging, coverage, or merge requirement is weakened by these operational changes.
