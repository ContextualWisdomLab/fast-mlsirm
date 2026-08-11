# Operation-specific ignored Rust subprocess deadlines

## Fixed

- Added bounded operation-specific deadlines for Cargo metadata, ignored-test inventory, and long-running statistical-study execution in the ignored Rust shard runner while retaining the independent GitHub Actions job ceiling.
- Timed-out POSIX child groups now receive bounded `SIGTERM`-to-`SIGKILL` cleanup, followed by a separately bounded final reap; if a killed process still blocks `communicate()`, inherited pipe handles are closed and only one additional bounded `wait()` is attempted before stable timeout evidence is returned.
- Timeout evidence omits command and captured child-output text, operator overrides remain constrained by per-operation minimum and maximum ranges, and deterministic cleanup/redaction/command-routing/cross-platform fallback contracts cover the migrated caller. Issue #555 remains open for the remaining repository subprocess operation classes.
