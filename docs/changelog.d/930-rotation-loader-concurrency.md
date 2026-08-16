# Secondary extension loader concurrency

## Fixed

- Serialize cache inspection and native initialization for the ATA, bifactor, multilevel, paired rating-range, and rotation secondary extension loaders so concurrent callers cannot observe temporary `sys.modules` entries before `exec_module()` completes.
- Preserve one-time shared-library loading, cached module identity, public loader APIs, and cleanup of failed initialization attempts without changing psychometric or numerical arithmetic.
