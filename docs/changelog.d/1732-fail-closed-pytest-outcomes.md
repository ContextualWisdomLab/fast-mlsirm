# Fail-closed pytest outcomes

## Fixed

- Escalate any pytest invocation with a skip, import-or-skip, skipif, xfail, or xpass outcome to a non-zero exit status via a session-level enforcement plugin, so a successful suite proves every collected evidence lane executed.
- Count collection-time skips as well as setup/call/teardown skips, and fail closed when outcome accounting cannot be observed.
- Review capability-gated non-executions through an explicit allowlist instead of rewriting capability-specific tests.
- Treat unexpected passes as failures by default with strict xfail handling.
