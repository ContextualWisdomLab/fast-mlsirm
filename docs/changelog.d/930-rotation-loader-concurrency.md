# Rotation extension loader concurrency

## Fixed

- Serialize secondary rotation-extension initialization so concurrent callers cannot observe the temporary `sys.modules` entry before native `exec_module()` completes.
- Preserve one-time shared-library loading, cached module identity, and cleanup of failed initialization attempts.
