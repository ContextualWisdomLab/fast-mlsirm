# Operation-specific ignored Rust subprocess deadlines

## Fixed

- Added bounded operation-specific deadlines for Cargo metadata, ignored-test inventory, and long-running statistical-study execution in the ignored Rust shard runner while retaining the independent GitHub Actions job ceiling.
- Timed-out POSIX child groups now receive bounded `SIGTERM`-to-`SIGKILL` cleanup and machine-readable timeout evidence that omits command and captured child-output text; operator overrides remain constrained by per-operation minimum and maximum ranges.
- Added deterministic configuration, cleanup, redaction, command-routing, and cross-platform fallback contracts. Issue #555 remains open for the remaining repository subprocess operation classes.
